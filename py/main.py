import argparse
from prepare_voc_yolo_dataset import prepare_data
from visualize import visualize
from yolo_model import train, eval, export

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def main():
    # Load the arguments parser and define the arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Pipeline Controller")
    parser.add_argument("--prepare", action="store_true", help="Prepare the dataset")
    parser.add_argument("--visualize", action="store_true", help="Visualize some sample images")
    parser.add_argument("--train", action="store_true", help="Train the YOLO model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or '0', '1', etc. for CUDA")
    parser.add_argument("--record", type=str, default="train", help="The record of the weights of a trained YOLO model, e.g. train, train2, train3, etc.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the YOLO model")
    parser.add_argument("--export", action="store_true", help="Export the trained model to ONNX")
    parser.add_argument("--all", action="store_true", help="Run the full training pipeline")
    args = parser.parse_args()

    # Print help if no main actions are specified
    if not (args.prepare or args.visualize or args.train or args.eval or args.export or args.all):
        parser.print_help()
        return

    # Run data preparation
    if args.all or args.prepare:
        print("\n=== Preparing the datasets ===")
        prepare_data("train")
        prepare_data("val")
        prepare_data("test")

    # Run visualization interactively over folder
    if args.all or args.visualize:
        print("\n=== Visualizing samples (press any key, 'q' to quit) ===")
        visualize("dataset/images/train", "dataset/labels/train", CLASSES)

    # Run training process
    if args.all or args.train:
        print("\n=== Training the YOLO Model ===")
        train(epochs=args.epochs, device=args.device)

    # Run evaluation process
    if args.all or args.eval:
        print("\n=== Evaluating the YOLO Model ===")
        eval(model_record=args.record)

    # Run export process
    if args.all or args.export:
        print("\n=== Exporting the YOLO Model ===")
        export()

if __name__ == "__main__":
    main()