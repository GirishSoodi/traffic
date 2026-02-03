from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to an image
img = "C:/Users/Girish/OneDrive/Desktop/WhatsApp Image 2025-12-01 at 14.01.19_f8234b52.jpg"

# Run inference multiple times to measure FPS accurately
num_runs = 100
start_time = time.time()
for _ in range(num_runs):
    results = model(img)
end_time = time.time()

total_time = end_time - start_time
fps = num_runs / total_time

print(f"Detection rate: {fps:.2f} FPS")
