# rl_control.py

import os
import sys
import random
import numpy as np
import pandas as pd
import pickle
import socket
import time
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# =========================
# CHOOSE CONTROL MODE HERE
# =========================
# "RL"    -> Q-learning adaptive control
# "FIXED" -> SUMO's built-in fixed-time program (no RL)
CONTROL_MODE = "RL"   # <-- change to "FIXED" or "RL" before running

# =========================
# TCP V2X Settings
# =========================
NS3_IP = "127.0.0.1"
NS3_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

V2X_ENABLED = False
try:
    sock.connect((NS3_IP, NS3_PORT))
    V2X_ENABLED = True
    print(f"[V2X] Connected to NS-3 at {NS3_IP}:{NS3_PORT}")
except Exception as e:
    print(f"[V2X] Could not connect to NS-3 ({e}). V2X disabled.")
    V2X_ENABLED = False

# =========================
# SUMO Setup
# =========================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

import traci

sumo_cfg = r"E:/Project/Project/sumo.env/simple.sumocfg"
tls_id = "clusterJ3_J4_J6_J7"
edges = ["E0", "E0.86", "-E0", "-E0.89", "E1", "-E1", "E2", "-E2"]
routes = [f"route_{i}" for i in range(12)]

df = pd.read_csv("traffic_timeseries.csv")
vehicle_counts = df['vehicle_count'].tolist()

# =========================
# Custom Attention Layer
# =========================
@register_keras_serializable()
class Attention(Layer):
    def build(self, input_shape):
        self.att_weight = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.att_bias = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        score = K.tanh(K.dot(x, self.att_weight) + self.att_bias)
        weights = K.softmax(score, axis=1)
        context = weights * x
        return K.sum(context, axis=1)

# =========================
# Load LSTM Model & Scaler
# =========================
model = load_model("traffic_lstm_attention.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# Q-Learning Setup
# =========================
q_table = {}
actions = [0, 1]      # 0 = keep current phase, 1 = switch to other direction
alpha = 0.1
gamma = 0.9
epsilon = 1.0         # start with full exploration
epsilon_min = 0.05
epsilon_decay = 0.97
MIN_GREEN_STEPS = 30  # 30 * 0.1s = 3 seconds min green
last_switch_step = -MIN_GREEN_STEPS

# Adjust mapping to your network directions if needed
DETECTOR_IDS = ["det_minus_E0", "det_E1", "det_E0", "det_minus_E2"]
# Assume:
#   det_minus_E0 + det_E1  -> North-South direction
#   det_E0 + det_minus_E2  -> East-West direction

# =========================
# Helper Functions
# =========================
def predict_lstm(history, seq_len=10):
    if len(history) < seq_len:
        seq = [history[-1]] * (seq_len - len(history)) + history
    else:
        seq = history[-seq_len:]
    seq_arr = np.array(seq).reshape(-1, 1)
    seq_scaled = scaler.transform(seq_arr).reshape(1, seq_len, 1)
    pred_scaled = model.predict(seq_scaled, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    return max(pred, 0)

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepHaltingNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def discretize_queue(q):
    if q < 3:
        return "Small"
    elif q < 7:
        return "Medium"
    else:
        return "Large"

def get_state():
    queue_counts = [get_queue_length(det) for det in DETECTOR_IDS]
    binned_queues = [discretize_queue(q) for q in queue_counts]
    phase = get_current_phase(tls_id)
    return tuple(binned_queues + [phase]), queue_counts

def get_reward(prev_queue, curr_queue):
    """
    Reward encourages:
      - Decrease in total queue (positive)
      - Penalizes large queues
    """
    delta = prev_queue - curr_queue  # >0 if queue decreased
    reward = 2.0 * delta - 0.1 * curr_queue
    return reward

def get_max_Q_value_of_state(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(actions))
    return np.max(q_table[state])

def get_action_from_policy(state):
    global epsilon
    # epsilon-greedy
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        if state not in q_table:
            q_table[state] = np.zeros(len(actions))
        return int(np.argmax(q_table[state]))

def apply_action(action, step):
    """
    RL signal control:
    action = 0 -> keep current phase
    action = 1 -> switch to the direction with higher demand
    """
    global last_switch_step

    if action == 0:
        return  # keep current phase

    # Enforce minimum green time
    if (step - last_switch_step) < MIN_GREEN_STEPS:
        return

    # Compute demand per direction
    q_vals = [get_queue_length(d) for d in DETECTOR_IDS]
    north_south = q_vals[0] + q_vals[1]  # adjust mapping if needed
    east_west = q_vals[2] + q_vals[3]

    current_phase = get_current_phase(tls_id)

    # You MUST match these indices with your SUMO tls logic!
    # Example assumption:
    #   phase 0 -> NS green, EW red
    #   phase 2 -> EW green, NS red
    if north_south >= east_west:
        target_phase = 0  # NS green
    else:
        target_phase = 2  # EW green

    if current_phase != target_phase:
        traci.trafficlight.setPhase(tls_id, target_phase)
        last_switch_step = step

def update_q_table(old_state, action, reward, new_state):
    if old_state not in q_table:
        q_table[old_state] = np.zeros(len(actions))
    old_q = q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    q_table[old_state][action] = old_q + alpha * (reward + gamma * best_future_q - old_q)

# =========================
# V2X: TCP Send to NS-3
# =========================
def send_positions_to_ns3(step):
    if not V2X_ENABLED:
        return

    try:
        veh_ids = traci.vehicle.getIDList()
    except Exception:
        veh_ids = []

    positions = []
    for vid in veh_ids:
        try:
            x, y = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
        except Exception:
            continue
        positions.append(f"{vid},{x:.2f},{y:.2f},{speed:.2f}")

    msg = f"{step}," + ";".join(positions)
    try:
        sock.sendall(msg.encode())
    except Exception as e:
        print(f"[V2X TX] send error: {e}")

# =========================
# Run SUMO + Control + V2X
# =========================
traci.start(["sumo-gui", "-c", sumo_cfg])
vehicle_id_counter = 0
history = []

TOTAL_STEPS = len(vehicle_counts)
cumulative_reward = 0.0
step_history = []
reward_history = []
queue_history = []

# Metrics for LSTM and control performance
y_true = []
y_pred = []
waiting_times = []
queue_lengths = []
stop_counts = []
throughput_counter = 0

print(f"=== Starting Simulation ({CONTROL_MODE} control) + V2X (TCP) ===")

# initial state for RL only
if CONTROL_MODE == "RL":
    # reset RL parameters for a fresh run
    q_table.clear()
    epsilon = 1.0
    last_switch_step = -MIN_GREEN_STEPS

    traci.simulationStep()
    state, _ = get_state()
    action = get_action_from_policy(state)
    prev_total_queue = 0
else:
    traci.simulationStep()
    state = None
    action = 0  # dummy
    prev_total_queue = 0

for step in range(TOTAL_STEPS):
    # =========================
    # Control Logic
    # =========================
    if CONTROL_MODE == "RL":
        apply_action(action, step)  # actively change phase
    # in FIXED mode we do NOT call apply_action -> SUMO runs its built-in fixed plan

    traci.simulationStep()
    send_positions_to_ns3(step)

    # --- LSTM prediction and logging ---
    history.append(vehicle_counts[step])
    pred_count = predict_lstm(history)
    y_true.append(vehicle_counts[step])
    y_pred.append(pred_count)

    # inject predicted number of vehicles
    for i in range(int(pred_count)):
        vid = f"v_{vehicle_id_counter}"
        route_id = routes[i % len(routes)]
        vehicle_id_counter += 1
        try:
            traci.vehicle.add(
                vehID=vid,
                typeID="car",
                routeID=route_id,
                depart=traci.simulation.getTime(),
                departLane="best"
            )
        except Exception:
            pass

    # --- State / reward / next action (only in RL mode) ---
    if CONTROL_MODE == "RL":
        new_state, raw_queues = get_state()
        total_queue = sum(raw_queues)

        reward = get_reward(prev_total_queue, total_queue)
        prev_total_queue = total_queue

        cumulative_reward += reward
        update_q_table(state, action, reward, new_state)
        state = new_state
        action = get_action_from_policy(state)

        # epsilon decay (slowly reduce exploration)
        if step % 200 == 0 and epsilon > epsilon_min:
            epsilon *= epsilon_decay

    else:
        # still need queue lengths, but no RL update
        _, raw_queues = get_state()
        total_queue = sum(raw_queues)
        reward = 0.0

    # --- Control metrics collection (common for both modes) ---
    veh_ids = traci.vehicle.getIDList()
    if veh_ids:
        step_wait_times = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
        waiting_times.append(np.mean(step_wait_times))
        step_stops = [traci.vehicle.getStopState(v) > 0 for v in veh_ids]
        stop_counts.append(np.sum(step_stops))
    else:
        waiting_times.append(0.0)
        stop_counts.append(0.0)

    queue_lengths.append(total_queue)
    throughput_counter = traci.simulation.getArrivedNumber()

    print(f"[{CONTROL_MODE}] Step={step:<4} Pred={pred_count:.2f} "
          f"Reward={reward:.2f} TotalQueue={total_queue}")

    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(total_queue)
    time.sleep(0.05)

# =========================
# Close SUMO & TCP
# =========================
traci.close()
sock.close()

print(f"\nTotal reward (only meaningful for RL): {cumulative_reward}")
print(f"Final Q-table size: {len(q_table)}")

# =========================
# Metrics Computation
# =========================
# ---- LSTM Metrics ----
y_true_arr = np.array(y_true)
y_pred_arr = np.array(y_pred)

mae = np.mean(np.abs(y_true_arr - y_pred_arr))
rmse = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))

nonzero_idx = y_true_arr != 0
if np.any(nonzero_idx):
    mpe = np.mean(
        np.abs((y_true_arr[nonzero_idx] - y_pred_arr[nonzero_idx]) /
               y_true_arr[nonzero_idx])
    ) * 100.0
    accuracy = 100.0 - mpe
else:
    accuracy = 0.0

if len(y_true_arr) > 0:
    threshold = np.percentile(y_true_arr, 90)
    peak_idx = np.where(y_true_arr >= threshold)[0]
    if len(peak_idx) > 0:
        peak_dev = np.mean(
            np.abs(y_true_arr[peak_idx] - y_pred_arr[peak_idx]) /
            y_true_arr[peak_idx]
        ) * 100.0
    else:
        peak_dev = 0.0
else:
    peak_dev = 0.0

print("\n=== LSTM Traffic Forecast Performance ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Prediction Accuracy: {accuracy:.2f}%")
print(f"Peak-hour Forecast Deviation: {peak_dev:.2f}%")

# ---- Control / RL Metrics ----
waiting_times_arr = np.array(waiting_times)
queue_lengths_arr = np.array(queue_lengths)
stop_counts_arr = np.array(stop_counts)

avg_waiting_time = np.mean(waiting_times_arr)
avg_queue_len = np.mean(queue_lengths_arr)
avg_stops = np.mean(stop_counts_arr)
throughput = throughput_counter

print(f"\n=== Traffic Control Performance ({CONTROL_MODE}) ===")
print(f"Average Waiting Time: {avg_waiting_time:.2f} sec")
print(f"Average Queue Length: {avg_queue_len:.2f} vehicles")
print(f"Average Number of Stops: {avg_stops:.2f} stops/vehicle")
print(f"Throughput: {throughput} vehicles/hour")

# =========================
# Export Metrics (and Q-table) to Excel
# =========================
excel_path = f"traffic_metrics_{CONTROL_MODE.lower()}.xlsx"

lstm_df = pd.DataFrame({
    "Metric": [
        "Mean Absolute Error (MAE)",
        "Root Mean Square Error (RMSE)",
        "Prediction Accuracy (%)",
        "Peak-hour Forecast Deviation (%)"
    ],
    "Value": [
        f"{mae:.2f}",
        f"{rmse:.2f}",
        f"{accuracy:.2f}",
        f"{peak_dev:.2f}"
    ]
})

control_df = pd.DataFrame([{
    "Mode": CONTROL_MODE,
    "Average Waiting Time (sec)": round(avg_waiting_time, 2),
    "Average Queue Length (vehicles)": round(avg_queue_len, 2),
    "Number of Stops per Vehicle": round(avg_stops, 2),
    "Throughput (vehicles/hour)": int(throughput)
}])

with pd.ExcelWriter(excel_path) as writer:
    lstm_df.to_excel(writer, sheet_name="LSTM_Metrics", index=False)
    control_df.to_excel(writer, sheet_name="Control_Metrics", index=False)

    # Export Q-table only for RL
    if CONTROL_MODE == "RL" and len(q_table) > 0:
        rows = []
        for state, q_vals in q_table.items():
            d0, d1, d2, d3, phase = state
            best_action = int(np.argmax(q_vals))
            rows.append({
                "det_minus_E0": d0,
                "det_E1": d1,
                "det_E0": d2,
                "det_minus_E2": d3,
                "phase": phase,
                "Q_keep": float(q_vals[0]),
                "Q_switch": float(q_vals[1]),
                "Best_Action": best_action,
                "Best_Q": float(np.max(q_vals))
            })
        q_df = pd.DataFrame(rows)
        q_df.to_excel(writer, sheet_name="Q_Table", index=False)

print(f"\nMetrics exported to Excel file: {excel_path}")

# =========================
# Visualization: Reward & Queue
# =========================
plt.figure(figsize=(12, 6))
plt.plot(step_history, reward_history, label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title(f"RL Training: Cumulative Reward over Steps ({CONTROL_MODE})")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(step_history, queue_history, label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length (Vehicles)")
plt.title(f"Total Queue Length over Steps ({CONTROL_MODE})")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# Visualization: Q-table Learning (for RL)
# =========================
if CONTROL_MODE == "RL" and len(q_table) > 0:
    # Build high-level heatmap: when does RL choose to SWITCH?
    cat2int = {"Small": 0, "Medium": 1, "Large": 2}
    heat = np.full((3, 3), np.nan)
    count = np.zeros((3, 3))

    for state, q_vals in q_table.items():
        d0, d1, d2, d3, phase = state
        best_action = int(np.argmax(q_vals))

        # Aggregate NS and EW load (max of their lanes)
        ns_level = max(cat2int[d0], cat2int[d1])  # NS direction
        ew_level = max(cat2int[d2], cat2int[d3])  # EW direction

        if np.isnan(heat[ns_level, ew_level]):
            heat[ns_level, ew_level] = 0.0
        heat[ns_level, ew_level] += best_action  # action 1 = switch
        count[ns_level, ew_level] += 1

    with np.errstate(invalid='ignore'):
        prob_switch = np.divide(
            heat,
            count,
            out=np.zeros_like(heat),
            where=count > 0
        )

    plt.figure(figsize=(6, 5))
    plt.imshow(prob_switch, origin='lower', aspect='auto')
    plt.colorbar(label="Probability of choosing SWITCH (action=1)")
    plt.xticks([0, 1, 2], ["Small", "Medium", "Large"])
    plt.yticks([0, 1, 2], ["Small", "Medium", "Large"])
    plt.xlabel("East-West Load Level")
    plt.ylabel("North-South Load Level")
    plt.title("RL Policy: When does it Switch?")
    plt.grid(False)
    plt.show()
