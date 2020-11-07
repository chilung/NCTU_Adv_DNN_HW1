import numpy as np
# import scipy.misc
import imageio
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE


class HW1():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        img_pd = pd.read_csv(os.path.join(self.root, 'images.csv'))
        label_pd = pd.read_csv(os.path.join(self.root, 'image_class_labels.csv'))
        train_test_pd = pd.read_csv(os.path.join(self.root, 'train_test_split.csv'))

        img_name_list = [img_fn for _, img_fn in img_pd.values.tolist()]
        # print(img_name_list)
        label_list = [label for _, label in label_pd.values.tolist()]
        # print(label_list)
        train_test_list = [split for _, split in train_test_pd.values.tolist()]
        # print(train_test_list)

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        # print(train_file_list)
        # print(test_file_list)

        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'training_data/training_data', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'training_data/training_data', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = HW1(root='./cs-t0828-2020-hw1')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = HW1(root='./cs-t0828-2020-hw1', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
