import cv2
import os

# Draws bounding boxes and labels on an image using OpenCV with keyboard navigation support
# Press any key to move to the next image; press 'q' to quit visualization
def visualize(image_dir, label_dir, class_name):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        if image is None or not os.path.exists(label_path):
            continue

        h, w = image.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                # extract the class id and normalized coordinates in YOLO format
                cls_id, x, y, bw, bh = map(float, line.split())
                # convert YOLO format back to pixels coords
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                # draw a green rentangle around the detected object, i.e. (0, 255, 0) is RGB for green
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # annotate and render the class name above the drawn box
                cv2.putText(image, class_name[int(cls_id)], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # show the image and waits indefinitely for the key Q to be pressed
        cv2.imshow("YOLO Annotated Image", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    # close all displayed windows
    cv2.destroyAllWindows()