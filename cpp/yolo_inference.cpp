#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

// Load class names from file
vector<string> load_class_list(const string& path) {
    vector<string> class_list;
    ifstream ifs(path);
    string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

int main() {
    // Load class names
    vector<string> class_names = load_class_list("../classes.txt");

    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Failed to open webcam!" << endl;
        return -1;
    }

    // ONNX Runtime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../best.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    // auto input_name = session.GetInputName(0, allocator);
    // auto output_name = session.GetOutputName(0, allocator);
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.4;

    // Real-time processing infinite-loop, will break when the key 'q' is pressed
    while (true) {
        Mat frame;
        cap >> frame;  // Capture a frame from webcam
        if (frame.empty()) break;

        // Preprocess, e.g. resize, normalize, convert to CHW format
        Mat resized;
        resize(frame, resized, Size(input_width, input_height));
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // normalize to [0,1]

        vector<Mat> chw(3);
        split(resized, chw);  // Convert HWC to CHW
        vector<float> input_tensor_values;
        for (int c = 0; c < 3; ++c)
            input_tensor_values.insert(input_tensor_values.end(), (float*)chw[c].datastart, (float*)chw[c].dataend);

        // Prepare input tensor
        array<int64_t, 4> input_shape = {1, 3, input_height, input_width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

        float* output = output_tensors.front().GetTensorMutableData<float>();
        auto type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        vector<int64_t> output_shape = type_info.GetShape();

        int num_detections = output_shape[1];
        int num_classes = class_names.size();

        // Postprocessing: filter and draw boxes
        for (int i = 0; i < num_detections; ++i) {
            float* det = output + i * (5 + num_classes);  // pointer to current detection
            float obj_conf = det[4];  // objectness confidence
            if (obj_conf < conf_threshold) continue;

            // Find the class with highest confidence
            float max_cls_score = 0.0f;
            int class_id = -1;
            for (int j = 5; j < 5 + num_classes; ++j) {
                if (det[j] > max_cls_score) {
                    max_cls_score = det[j];
                    class_id = j - 5;
                }
            }

            // Final confidence threshold check
            if (max_cls_score > conf_threshold) {
                // Convert normalized coordinates to pixel values
                float cx = det[0] * frame.cols;
                float cy = det[1] * frame.rows;
                float w = det[2] * frame.cols;
                float h = det[3] * frame.rows;
                int left = static_cast<int>(cx - w / 2);
                int top = static_cast<int>(cy - h / 2);

                // Draw bounding box and class label
                rectangle(frame, Rect(left, top, static_cast<int>(w), static_cast<int>(h)), Scalar(0, 255, 0), 2);
                putText(frame, format("%s: %.2f", class_names[class_id].c_str(), max_cls_score), Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }

        // Display the annotated frame
        imshow("YOLOv8 Real-Time Detection (ONNX Runtime)", frame);

        // Exit on 'q' key press
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
