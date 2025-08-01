from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train2/weights/best.pt")  # adjust if path is different

# Run prediction on your sample image
results = model.predict(source="test2.png", save=True, conf=0.1)

# Output folder will be runs/detect/predict
print("✅ Prediction complete! Check the 'runs/detect/predict' folder.")
