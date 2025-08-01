from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='C:/Users/HP/Downloads/dataset/Output/2025-07-31-09-50-20/yolo_params.yaml',
    epochs=20,
    imgsz=640,
    batch=8
)
