cmake_minimum_required(VERSION 3.10)
project(YOLOv8_ONNX_Runtime)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Executable target
add_executable(yolo_inference yolo_inference.cpp)
target_link_libraries(yolo_inference ${OpenCV_LIBS} onnxruntime)