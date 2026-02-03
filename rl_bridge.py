# bridge.py
import traci
import socket
import time
import threading

NS3_HOST = "127.0.0.1"
NS3_PORT = 5555

SUMO_BINARY = "sumo"   # or "sumo-gui"
SUMO_CFG = "simple.sumocfg"
STEP_LENGTH = 1.0  # simulation step length (seconds)

# Connect to ns-3 server (TCP)
def connect_ns3(host, port, retries=10, wait=0.5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for i in range(retries):
        try:
            s.connect((host, port))
            print("Connected to ns-3 at", host, port)
            s.settimeout(2.0)
            return s
        except Exception as e:
            print("ns-3 not ready yet, retrying...", e)
            time.sleep(wait)
    raise RuntimeError("Failed to connect to ns-3")

def send_msg(sock, msg):
    try:
        sock.sendall(msg.encode())
        # optionally read ack
        try:
            ack = sock.recv(4096).decode()
            return ack
        except socket.timeout:
            return None
    except Exception as e:
        print("Send failed:", e)
        return None

def main():
    # start SUMO and TraCI
    sumo_cmd = [SUMO_BINARY, "-c", SUMO_CFG, "--step-length", str(STEP_LENGTH)]
    traci.start(sumo_cmd)
    print("SUMO started")

    # connect to ns-3
    sock = connect_ns3(NS3_HOST, NS3_PORT)

    current_vehicles = set()

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()  # advance SUMO by STEP_LENGTH
            step += 1

            vehicles = traci.vehicle.getIDList()
            vehicles_set = set(vehicles)

            # detect removals
            removed = current_vehicles - vehicles_set
            for vid in removed:
                msg = f"REMOVE,{vid}\n"
                send_msg(sock, msg)

            # detect new spawns
            new = vehicles_set - current_vehicles
            for vid in new:
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                # convert SUMO coordinates -> ns-3 coordinates if necessary (here we use same coords)
                msg = f"SPAWN,{vid},{x},{y},0.0,{speed},{step}\n"
                send_msg(sock, msg)

            # update all vehicles
            for vid in vehicles:
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                msg = f"UPDATE,{vid},{x},{y},0.0,{speed},{step}\n"
                send_msg(sock, msg)

            current_vehicles = vehicles_set

            # Optionally, sleep to keep real-time-ish pace:
            # time.sleep(STEP_LENGTH * 0.01)  # small sleep if needed

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        traci.close()
        sock.close()
        print("Finished bridge")

if __name__ == "__main__":
    main()
