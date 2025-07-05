# Real-Time Object Detection

## 🧠 Overview
### Project Goal
This project demonstrates a complete pipeline for real-time object detection using a custom-trained YOLOv8 model in Python and high-performance inference using OpenCV in C++.

### Project Outline
- **Dataset**: PASCAL VOC 2007 and 2012
- **Model Training**: YOLOv8 (Ultralytics) in Python
- **Model Format**: ONNX (Open Neural Network Exchange) for cross-platform deployment
- **Deployment**: Real-time webcam detection in C++ using OpenCV DNN module
- **Purpose**: Learn and integrate modern CV models with real-time system-level programming

## 📦 Dataset Setup: Pascal VOC 2007 + 2012
Because the combined dataset is large, it’s not included in this repo. Follow these steps to download:
1. Download VOC2007 and VOC2012: you can manually download them from Kaggle (see the links in the Acknowledgement [below](https://github.com/NathanDang97/Real-Time-Object-Detection/edit/main/README.md#-acknowledgments))
2. After downloading, extract both .zip files into this folder structure: ensure that each ImageSets/Main/ folder contains the train.txt, val.txt, and test.txt files.
```
VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── JPEGImages/
│   └── ImageSets/
├── VOC2012/
│   ├── Annotations/
│   ├── JPEGImages/
│   └── ImageSets/
```

## 🔧 Usage
All training and evaluation is done via Python using Ultralytics' YOLOv8 interface. The training and evaluation scripts can be found in the _py_ folder. Run the steps using the following commands.
1. Prepare the dataset
```bash
python main.py --prepare
```
2. Visualize Annotations (Optional)
```bash
python main.py --visualize
```
3. Train the model
```bash
# for default parameters, e.g. no. epochs = 30, device = "cpu", selected model = "train":
python main.py --train
# or customized parameters, e.g. no. epochs = 100, device = "0" (representing GPU if supported), selected model = "train2":
python main.py --train --epochs 100 --device "0" --record "train2"
```
4. Evaluate the model
```bash
python main.py --eval
```
5. Export the model
```bash
python main.py --export
```

After training, you can run the real-time C++ inference (ONNX Runtime) using a webcam. The source code and be found in the _cpp_ folder. The steps are below.
1. Create the build folder (Optional, but keeps the project clean)
```bash
mkdir build && cd build
```
2. Compile the C++ file using CMake. Make sure ONNX Runtime is installed in your system. ONNX Runtime can be downloaded [here](https://github.com/microsoft/onnxruntime/releases) Use the CMakeLists.txt in the _cpp_ folder assuming you extract the _onnxruntime_ folder inside your working directory (thus, change the directory path if not).
```bash
cmake ..
make
```
3. Make sure _yolo_inference.cpp_ is executable and Run
```bash
chmod +x yolo_inference
./yolo_inference
```
4. To exist, press the key 'Q'.

## 📝 Acknowledgments
- Dataset: [PASCAL VOC 2007](https://www.kaggle.com/datasets/zaraks/pascal-voc-2007/data), [PASCAL VOC 2012](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)
- Model Framework: [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- Inference Engine: [OpenCV DNN Module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
