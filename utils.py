import os
import cv2
import xml.etree.ElementTree as ET

import imutils
import numpy as np
from tqdm import tqdm

def yolo_to_voc(yolo_txt_path, image_name, class_list, img_width=3208, img_height=2200, img_depth=3):
    with open(yolo_txt_path, 'r') as file:
        lines = file.readlines()

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "img"
    ET.SubElement(annotation, "filename").text = image_name
    ET.SubElement(annotation, "path").text = os.path.join('images', image_name)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for line in lines:
        parts = line.strip().split()
        cls, x_center, y_center, width, height = int(parts[0]), float(parts[1]), float(parts[2]), float(
            parts[3]), float(parts[4])

        xmin = int((x_center - width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        xmax = int((x_center + width / 2) * img_width)
        ymax = int((y_center + height / 2) * img_height)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_list[cls]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    return ET.tostring(annotation).decode()


def process_directory(label_dir, class_list):
    xml_dir = os.path.join(label_dir, 'xml')
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    for filename in tqdm(os.listdir(label_dir), desc='Converting files...'):
        if filename.endswith('.txt'):
            yolo_path = os.path.join(label_dir, filename)
            image_name = filename.replace('.txt', '.jpg')  # Assuming images are in JPG format

            xml_output = yolo_to_voc(yolo_path, image_name, class_list)

            with open(os.path.join(xml_dir, filename.replace('.txt', '.xml')), 'w') as output_xml:
                output_xml.write(xml_output)

def convert_yolo_dir():
    label_directory = 'data/labels_8cls'
    classes = ['PA-1', 'PA-2', 'PA-3', 'PA-4', 'PA-5', 'SPA-1', 'SPA-2', 'SPA-3']
    process_directory(label_directory, classes)

# functions to visualise annotations
def generate_colors(num_classes):
    colors = []
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]

    for hsv in hsv_tuples:
        hsv_in_numpy = np.uint8([[[hsv[0] * 255, hsv[1] * 255, hsv[2] * 255]]])
        bgr = cv2.cvtColor(hsv_in_numpy, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))

    return colors

def read_yolo_annotations(filepath, img_width, img_height):
    boxes = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, x_center, y_center, width, height = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)
            boxes.append((cls, xmin, ymin, xmax, ymax))
    return boxes

def read_voc_annotations(filepath, classes):
    boxes = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        cls = classes.index(name)  # Get index from the classes list
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((cls, xmin, ymin, xmax, ymax))

    return boxes

def visualise_annotations(img_path, boxes, classes, COLORS):
    image = cv2.imread(img_path)
    for box in boxes:
        cls, xmin, ymin, xmax, ymax = box
        color = COLORS[cls]
        label = classes[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    resized = imutils.resize(image, width=1000)
    cv2.imshow('Check Annotations', resized)
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
        elif key == 32:
            cv2.destroyAllWindows()
            break


def visualise(image_dir, label_dir, classes=None, img_width=3208, img_height=2200):
    if classes is None:
        classes = ['PA-1', 'PA-2', 'PA-3', 'PA-4', 'PA-5', 'SPA-1', 'SPA-2', 'SPA-3']

    COLORS = generate_colors(len(classes))

    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            yolo_path = os.path.join(label_dir, filename)
            img_path = os.path.join(image_dir, filename.replace('.txt', '.jpg'))
            boxes = read_yolo_annotations(yolo_path, img_width, img_height)
            visualise_annotations(img_path, boxes, classes, COLORS)

        elif filename.endswith('.xml'):
            xml_path = os.path.join(label_dir, filename)
            img_path = os.path.join(image_dir, filename.replace('.xml', '.jpg'))
            boxes = read_voc_annotations(xml_path, classes)
            visualise_annotations(img_path, boxes, classes, COLORS)


def check_images_labels(label_dir, image_dir):
    '''
    checks for presence of all images/labels in respective directories
    :param image_directory: directory to images
    :param label_directory: directory to labels
    '''
    image_files = set(os.listdir(image_dir))
    annotation_files = set(os.listdir(label_dir))
    image_errors = 0
    label_errors = 0

    images_checked = 0
    labels_checked = 0

    for image in image_files:
        labels_checked += 1
        txt_annotation = image.replace('.jpg', '.txt').replace('.png', '.txt')
        xml_annotation = image.replace('.jpg', '.xml').replace('.png', '.xml')

        if txt_annotation not in annotation_files and xml_annotation not in annotation_files:
            label_errors += 1
            print(f"[ERROR]: Missing annotation for image {image}")

    for annotation in annotation_files:
        images_checked += 1
        corresponding_image_jpg = annotation.replace('.txt', '.jpg').replace('.xml', '.jpg')
        corresponding_image_png = annotation.replace('.txt', '.png').replace('.xml', '.png')

        if corresponding_image_jpg not in image_files and corresponding_image_png not in image_files:
            image_errors += 1
            print(f"[ERROR]: Missing image for annotation {annotation}")

    print(f'[INFO] Images checked: {images_checked}\n[INFO] Labels checked: {labels_checked}\n[INFO] Errors: {label_errors} labels, {image_errors} images\n')
if __name__ == "__main__":
    label_dir = r'data/xml'
    image_dir = r'data/images'

    check_images_labels(label_dir=label_dir,
                        image_dir=image_dir)

    classes = ['PA-1', 'PA-2', 'PA-3', 'PA-4', 'PA-5', 'SPA-1', 'SPA-2', 'SPA-3']

    visualise(label_dir=label_dir,
              image_dir=image_dir,
              classes=classes)


