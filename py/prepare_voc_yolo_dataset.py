import os
import shutil
import xml.etree.ElementTree as ET

# List of class names from the Pascal VOC dataset
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Paths to the VOC2007 and VOC2012 datasets
# Note:
# - change this depends on the project structure
# - update this dictionary if more data are collected later
VOC_ROOTS = {
    "VOC2007": "../VOCDevKit/VOC2007",
    "VOC2012": "../VOCDevKit/VOC2012"
}

OUTPUT_DIR = "dataset"

# Convert VOC bounding box (xmin, xmax, ymin, ymax) to YOLO format (x_center, y_center, w, h)
# All values are normalized by the image width and height
def convert_box(size, box):
    # compute the normalizing scales
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    # compute the center of the box
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    # compute the width and height of the box
    w = box[1] - box[0]
    h = box[3] - box[2]
    # return the normalized x_center, y_center, width, and height
    return (x_center * dw, y_center * dh, w * dw, h * dh)

# Parses a single Pascal VOC XML file and writes YOLO-format annotations to a .txt file.
# Each line in the output file represents a bounding box in the format:
# class_id, x_center, y_center, width, and height (all normalized between 0 and 1)
def convert_annotation(xml_path, out_path):
    # parse the .xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    with open(out_path, 'w') as out_file:
        for obj in root.iter('object'):
            # filter out objects labeled as "difficult"
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            cls = obj.find('name').text
            # filter out objects not in the CLASSES list
            if cls not in CLASSES:
                continue
            cls_id = CLASSES.index(cls)
            # extract the <bnbbox> elements as an argument for the convert_box method
            xml_box = obj.find('bndbox')
            box = (
                float(xml_box.find('xmin').text),
                float(xml_box.find('xmax').text),
                float(xml_box.find('ymin').text),
                float(xml_box.find('ymax').text)
            )
            # convert into YOLO format (class_id x_center y_center w h) and write to .txt file
            voc_bounding_box = convert_box((w, h), box)
            out_file.write(f"{cls_id} " + " ".join(map(str, voc_bounding_box)) + "\n")

# Process either the 'train', 'val', or 'test' split for both VOC2007 and VOC2012 datasets.
# Convert all XML annotations to YOLO format and copies the associated images to
# a structured directory under dataset/images/ and dataset/labels/.
def prepare_data(type):
    print(f"\nPreparing {type} set...")
    # read the corresponding .txt file defined by the argument type
    img_out_dir = os.path.join(OUTPUT_DIR, "images", type)
    lbl_out_dir = os.path.join(OUTPUT_DIR, "labels", type)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    images_count = 0
    for voc_name, voc_path in VOC_ROOTS.items():
        # skip the invalid/corrupted data file
        txt_path = os.path.join(voc_path, "ImageSets", f"{type}.txt")
        if not os.path.exists(txt_path):
            print(f"Skipping {voc_name}: no {type}.txt found!")
            continue

        with open(txt_path) as f:
            image_ids = f.read().strip().split()

        for image_id in image_ids:
            # copy the corresponding .jpg file to the dataset/images/{type} folder
            image_src = os.path.join(voc_path, "JPEGImages", f"{image_id}.jpg")
            # prefix filenames with the dataset name to avoid collision
            image_dst = os.path.join(img_out_dir, f"{voc_name}_{image_id}.jpg")
            # convert the corresponding .xml to .txt in dataset/labels/{type} folder
            xml_src = os.path.join(voc_path, "Annotations", f"{image_id}.xml")
            # prefix filenames with the dataset name to avoid collision
            lbl_dst = os.path.join(lbl_out_dir, f"{voc_name}_{image_id}.txt")

            if not os.path.exists(image_src) or not os.path.exists(xml_src):
                continue
            
            # save the images and convert the .xml into YOLO format
            shutil.copy2(image_src, image_dst)
            convert_annotation(xml_src, lbl_dst)
            images_count += 1

    print(f"{type} set is ready with {images_count} images.")