from ultralytics import YOLO

# Initialize and trains a YOLOv8 model using the Ultralytics interface with the specified config.
# Apply transfer learning to help converge faster
def train(epochs=10, device='cpu'):
    # load the pre-trained YOLOv8n model config
    model = YOLO("yolov8n.pt")
    # run training
    model.train(data="data.yaml", epochs=epochs, imgsz=416, batch=8, device=device)

# Evaluate a trained YOLOv8 model
def eval(model_record):
    # load the trained weights
    model = YOLO(f"runs/detect/{model_record}/weights/best.pt")
    if model is None:
        print("Model not found! Double check the specified path to the model!")
        return
    # evaluate on the test set
    model.val(data="data.yaml", split="test")

# Load a trained YOLOv8 model and exports it to the ONNX format for deployment or inference in other frameworks.
def export():
    # load the trained weights
    model = YOLO("runs/detect/train/weights/best.pt")
    # export to ONNX format for inference later
    model.export(format="onnx", dynamic=True)