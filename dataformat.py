# coding: utf-8
import os
import shutil
import json
import xml.etree.ElementTree as ET

import cv2
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch


dataset_path = "./datasets/KitchenMice-v1"


def classify_images_labels(
        dataset_path: str | os.PathLike,
        image_fmt: tuple | list = [".png", ".jpeg", ".jpg", ".gif"], 
        label_fmt: str = ".json"
    ):
    images_path = os.path.join(os.path.abspath(dataset_path), "images")
    labels_path = os.path.join(os.path.abspath(dataset_path), f"{label_fmt.replace(".", "")}_labels")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for root, _, files in tqdm(os.walk(dataset_path)):
        if root == dataset_path:
            for file in files:
                if file.endswith(tuple(image_fmt)):
                    shutil.move(os.path.join(root, file), os.path.join(images_path, file))
                elif file.endswith(label_fmt):
                    shutil.move(os.path.join(root, file), os.path.join(labels_path, file))
            break

    print(Fore.GREEN + f"Images are moved to {images_path}")
    print(Fore.GREEN + f"Labels are moved to {labels_path}")
    print(Style.RESET_ALL)

    return images_path, labels_path


def get_file_list_map(dir: str | os.PathLike) -> dict:
    files_map = dict(
        (os.path.splitext(file)[0], os.path.join(root, file)) 
        for root, _, files in os.walk(dir) 
        for file in files
        if not file.startswith(".")
    )
    return files_map


def get_file_index_map(dir: str | os.PathLike) -> dict:
    index_map, index_image_map = dict(), dict()
    i = 1
    for _, _, files in os.walk(dir):
        for file in files:
            if file.startswith("."):
                continue
            index = str(i).zfill(6).replace(" ", "")
            index_map[os.path.splitext(file)[0]] = f"index-{index}"
            index_image_map[f"index-{index}"] = os.path.join(file)
            i += 1
    return index_map, index_image_map


index_map, index_image_map = get_file_index_map("./datasets/KitchenMice-v1/images")
reversed_index_map = {v: k for k, v in index_map.items()}


def json_to_yolov8(
        images_dir: str | os.PathLike,
        labels_dir: str | os.PathLike,
    ):
    """Convert object labels to YOLOv8"""
    output_dir = os.path.join(f"{labels_dir}/../", "txt_labels")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_map = get_file_list_map(images_dir)
    
    global index_map

    for root, _, files in tqdm(os.walk(labels_dir)):
        if root == labels_dir:
            for file in files:
                if file.startswith("."):
                    continue

                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    labels = data["labels"]  # list[dict]

                image = cv2.imread(images_map[file.replace(".json", "")])
                height, width, _ = image.shape

                label_file = os.path.join(output_dir, f"{index_map[file.replace(".json", "")]}.txt")
                fp = open(label_file, 'w')
                for label in labels:
                    cx = (label['x1'] + label['x2']) / 2
                    cy = (label['y1'] + label['y2']) / 2
                    w = label['x2'] - label['x1']
                    h = label['y2'] - label['y1']

                    cx /= width
                    cy /= height
                    w /= width
                    h /= height

                    fp.write(f"0 {cx} {cy} {w} {h}\n")

                fp.close()
            break

    print(Fore.GREEN + f"Labels converted to YOLOv8 format and saved to {output_dir}.")
    print(Style.RESET_ALL)


def yolov8_to_voc(
        yolov8_labels_path: str | os.PathLike,
        voc_labels_path: str | os.PathLike,
        images_path: str | os.PathLike
    ):
    """Convert YOLOv8 to VOC format"""
    if not os.path.exists(voc_labels_path):
        os.makedirs(voc_labels_path)

    images_map = get_file_list_map(images_path)

    global index_map, index_image_map

    for label_file in tqdm(os.listdir(yolov8_labels_path)):
        if label_file.startswith("."):
            continue
        
        yolo_file = os.path.join(yolov8_labels_path, label_file)
        voc_file = os.path.join(voc_labels_path, label_file.replace(".txt", ".xml"))
        image_file = os.path.join(images_path, index_image_map[label_file.replace(".txt", "")])

        height, width, _ = cv2.imread(image_file).shape

        tree = ET.ElementTree()
        root = ET.Element('annotation')
        tree._setroot(root)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'

        fp = open(yolo_file, 'r')
        for line in fp.readlines():
            parts = line.strip().split(" ")
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x_min = int((cx - w / 2) * width)
            y_min = int((cy - h / 2) * height)
            x_max = int((cx + w / 2) * width)
            y_max = int((cy + h / 2) * height)

            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = "mouse" # str(class_id)
            ET.SubElement(obj, 'bndbox')
            ET.SubElement(obj.find('bndbox'), 'xmin').text = str(x_min)
            ET.SubElement(obj.find('bndbox'), 'ymin').text = str(y_min)
            ET.SubElement(obj.find('bndbox'), 'xmax').text = str(x_max)
            ET.SubElement(obj.find('bndbox'), 'ymax').text = str(y_max)

        tree.write(voc_file)
        fp.close()

    print(Fore.GREEN + f"Labels converted to VOC format and saved to {voc_labels_path}.")
    print(Style.RESET_ALL)


def split_train_test_val(
        images_path: str | os.PathLike,
        train_size: float = 0.8,
        test_size: float = 0.1,
        valid_size: float = 0.1,
    ) -> tuple[list]:
    """Split dataset into train, test and validation set"""

    images_list = list(get_file_list_map(images_path).values())

    # split images to train, test and valid
    train_images, valid_images = train_test_split(
        images_list, train_size=train_size, test_size=valid_size + test_size, random_state=42
    )
    test_images, valid_images = train_test_split(
        valid_images, test_size=test_size / (test_size + valid_size), random_state=42
    )

    return train_images, test_images, valid_images


def copy_train_test_val(
        images_path: str | os.PathLike,
        # labels_path: str | os.PathLike,
        sub_dir: str | os.PathLike = "YOLOv8",
        train_size: float = 0.8,
        test_size: float = 0.1,
        valid_size: float = 0.1,
    ):
    """Split dataset into train, test and validation set"""
    base_path = os.path.abspath(os.path.dirname(images_path))

    train_path = os.path.join(base_path, sub_dir, "train")
    test_path = os.path.join(base_path, sub_dir, "test")
    valid_path = os.path.join(base_path, sub_dir, "valid")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    train_images_path = os.path.join(train_path, "images")
    train_labels_path = os.path.join(train_path, "labels")
    test_images_path = os.path.join(test_path, "images")
    test_labels_path = os.path.join(test_path, "labels")
    valid_images_path = os.path.join(valid_path, "images")
    valid_labels_path = os.path.join(valid_path, "labels")
    if not os.path.exists(train_images_path):
        os.makedirs(train_images_path)
    if not os.path.exists(train_labels_path):
        os.makedirs(train_labels_path)
    if not os.path.exists(test_images_path):
        os.makedirs(test_images_path)
    if not os.path.exists(test_labels_path):
        os.makedirs(test_labels_path)
    if not os.path.exists(valid_images_path):
        os.makedirs(valid_images_path)
    if not os.path.exists(valid_labels_path):
        os.makedirs(valid_labels_path)

    train_images, test_images, valid_images = split_train_test_val(
        images_path=images_path, train_size=train_size, test_size=test_size, valid_size=valid_size
    )

    for image in train_images:
        shutil.copy(image, os.path.join(train_images_path))
        shutil.copy(os.path.splitext(image.replace("jpg_images", "txt_labels"))[0] + ".txt", os.path.join(train_labels_path))

    for image in test_images:
        shutil.copy(image, os.path.join(test_images_path))
        shutil.copy(os.path.splitext(image.replace("jpg_images", "txt_labels"))[0] + ".txt", os.path.join(test_labels_path))

    for image in valid_images:
        shutil.copy(image, os.path.join(valid_images_path))
        shutil.copy(os.path.splitext(image.replace("jpg_images", "txt_labels"))[0] + ".txt", os.path.join(valid_labels_path))

    print(Fore.GREEN + f"Images split into train {train_size}, validation {valid_size} and test {test_size}.")
    print(Fore.GREEN + f"Train dataset: {train_path}")
    print(Fore.GREEN + f"Test dataset: {test_path}")
    print(Fore.GREEN + f"Validation dataset: {valid_path}")
    print(Style.RESET_ALL)


def image_to_jpg(
        images_path: str | os.PathLike,
        output_path: str | os.PathLike = None,
    ):
    """Convert image files to jpg"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    global index_map

    for root, _, files in tqdm(os.walk(images_path)):
        if root == images_path:
            for file in files:
                image = cv2.imread(os.path.join(root, file))
                file_name, ext = os.path.splitext(file)
                if ext.lower() != ".jpg":
                    cv2.imwrite(os.path.join(output_path, f"{index_map[file_name]}.jpg"), image)
                else:
                    shutil.copy(os.path.join(root, file), os.path.join(output_path, f"{index_map[file_name]}.jpg"))
            break

    print(Fore.GREEN + f"Images converted to .jpg format and saved to {output_path}.")
    print(Style.RESET_ALL)


if __name__ == "__main__":
    # _, labels_path = classify_images_labels(dataset_path)
    # labels_path = json_to_yolov8("./datasets/KitchenMice-v1/images", "./datasets/KitchenMice-v1/json_labels")
    copy_train_test_val("./datasets/KitchenMice-v1/jpg_images")
    # yolov8_to_voc("./datasets/KitchenMice-v1/txt_labels", "./datasets/KitchenMice-v1/xml_labels", "./datasets/KitchenMice-v1/images")
    # image_to_jpg("./datasets/KitchenMice-v1/images", "./datasets/KitchenMice-v1/jpg_images")
