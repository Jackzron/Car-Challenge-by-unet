import csv
import os


class csvReader():

    @staticmethod
    def generator_with_csv(path):
        """
        用于返回图片与标签对应的地址
        :param path: csv文件位置
        :return: 生成器：训练图片地址、标签（地址）
        """
        csvFile = open(path, "r")
        reader = csv.reader(csvFile)
        content = list(reader)
        for item in content:
            yield item[0], item[1]

    @staticmethod
    def make_csv(trainSet, labelSet, save_file_location):
        rows = []
        for x, y in zip(trainSet, labelSet):
            one_row = (x, y)
            rows.append(one_row)
        if '.' not in save_file_location:
            save_file_location = "csv"
        with open(save_file_location, 'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerows(rows)

    @staticmethod
    def trainGenerator(batch_size, train_path, label_path, aug_dict, image_folder=None, mask_folder=None,
                       image_color_mode="grayscale",
                       mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                       flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(512, 512), seed=1):
        from keras_preprocessing.image import ImageDataGenerator
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_directory(
            train_path,
            # classes=[image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        mask_generator = mask_datagen.flow_from_directory(
            label_path,
            # classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed)
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            # img, mask = adjustData(img, mask, flag_multi_class, num_class)
            yield (img, mask)


"data/membrane/train/image/0.png"
if __name__ == '__main__':
    path_train = r"E:\train_data\carChallenge\train_hq"
    train_dirs = os.listdir(path_train)
    train_dirs_with_path = []
    for item in train_dirs:
        train_dirs_with_path.append(os.path.join(path_train, item))

    path_label = r"E:\train_data\carChallenge\train_masksssssss"
    lable_dirs = os.listdir(path_label)
    lable_dirs_with_path = []
    for item in lable_dirs:
        lable_dirs_with_path.append(os.path.join(path_label, item))

    csvReader.make_csv(train_dirs_with_path, lable_dirs_with_path, r"E:\train_data\carChallenge\mycsv.csv")
