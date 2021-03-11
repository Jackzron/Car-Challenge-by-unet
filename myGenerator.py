import csv
import json
import os
import numpy as np
import cv2
from matplotlib import pyplot


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    return cv_img


def show_array_to_image(arrayData):
    pyplot.imshow(arrayData)
    pyplot.show()


def load_train(csvDir, width, height, batch_size):
    fx = 0.0
    fy = 0.0
    # 处理列表得到数组
    images_path = []
    labels_path = []
    csvFile = open(csvDir, "r")
    reader = csv.reader(csvFile)
    content = list(reader)
    for item in content:
        images_path.append(item[0])
        labels_path.append(item[1])
    # 进入循环读取照片
    # print(len(images_path))
    # for image, label in zip(images_path, labels_path):
    while True:
        image_data_array = []
        label_data_array = []
        index_group = np.random.randint(0, len(images_path), batch_size)
        # print("batch_size:", str(index_group))
        for index in index_group:
            image = images_path[index]
            label = labels_path[index]

            image_data = cv_imread(image)
            image_data = cv2.resize(image_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            image_data = image_data.astype(np.float32)
            image_data = np.multiply(image_data, 1.0 / 255.0)
            image_data_array.append(image_data)

            label_data = cv_imread(label)
            # label_data = cv2.cvtColor(label_data, cv2.COLOR_GRAY2BGR) # 颜色转化
            label_data = cv2.resize(label_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            label_data = label_data.astype(np.float32)
            label_data = np.multiply(label_data, 1.0 / 255.0)
            label_data_array.append(label_data)

        image_data_r = np.array(image_data_array)
        label_data_r = np.array(label_data_array)

        yield image_data_r, label_data_r
        # image_data_array.append(image_data)
        # label_data_array.append(label_data)
        # return image_data_array, label_data_array


def load_test(csvDir, width, height, batch_size):
    fx = 0.0
    fy = 0.0
    # 处理列表得到数组
    images_path = []
    labels_path = []
    csvFile = open(csvDir, "r")
    reader = csv.reader(csvFile)
    content = list(reader)
    for item in content:
        images_path.append(item[0])
        labels_path.append(item[1])
    # 进入循环读取照片

    # for image, label in zip(images_path, labels_path):
    image_data_array = []
    label_data_array = []
    index_group = np.random.randint(0, len(images_path), batch_size)
    # print("batch_size:", str(index_group))
    for index in index_group:
        image = images_path[index]
        label = labels_path[index]

        image_data = cv_imread(image)
        image_data = cv2.resize(image_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        image_data = image_data.astype(np.float32)
        image_data = np.multiply(image_data, 1.0 / 255.0)
        image_data_array.append(image_data)

        label_data = cv_imread(label)
        # label_data = cv2.cvtColor(label_data, cv2.COLOR_GRAY2BGR) # 颜色转化
        label_data = cv2.resize(label_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        label_data = label_data.astype(np.float32)
        label_data = np.multiply(label_data, 1.0 / 255.0)
        label_data_array.append(label_data)

    image_data_r = np.array(image_data_array)
    label_data_r = np.array(label_data_array)

    return image_data_r, label_data_r
    # image_data_array.append(image_data)
    # label_data_array.append(label_data)
    # return image_data_array, label_data_array


def load_validate(validate_path, width, height):
    root = os.getcwd()
    with open(validate_path, 'r') as load_f:
        load_dict = json.load(load_f)

        # num_image = len(load_dict)
        # 只产生512个数据，避免内存过大
        while True:
            images = []
            labels = []
            classes = np.zeros(61)
            number = np.random.random_integers(0, len(load_dict) - 1, 512)

            for image in number:
                index = load_dict[image]["disease_class"]
                path = load_dict[image]['image_id']
                img_path = os.path.join(root, 'AgriculturalDisease_validationset', 'images', path)
                image_data = cv_imread(img_path)
                image_data = cv2.resize(image_data, (width, height), 0, 0, cv2.INTER_LINEAR)
                image_data = image_data.astype(np.float32)
                image_data = np.multiply(image_data, 1.0 / 255.0)

                images.append(image_data)
                label = np.zeros(len(classes))
                label[index] = 1
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)

            yield images, labels


if __name__ == '__main__':
    lt = load_train(r"E:\train_data\carChallenge\mycsv.csv", 512, 512)
    # print(next(lt)[0].shape)

    # 将图片转化为jpg格式
    # images_path = []
    # labels_path = []
    # csvFile = open(r"E:\train_data\carChallenge\mycsv.csv", "r")
    # reader = csv.reader(csvFile)
    # content = list(reader)
    # for item in content:
    #     labels_path_one =item[1]
    #     # print(type(labels_path_one))
    #     name = labels_path_one.split("\\")
    #     # print(name[4])
    #     name = name[4].split(".")
    #     name = name[0]+".jpg"
    #     url = r"E:\train_data\carChallenge\train_maskss"
    #     path = os.path.join(url, name)
    #     from PIL import Image
    #     im = Image.open(labels_path_one)
    #     im = im.convert('RGB')
    #     im.save(path)
