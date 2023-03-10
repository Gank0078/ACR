import logging
import math

import os
import sys
import pickle

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
import torchvision
import torch.utils.data as data

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

SVHN_mean = (0.4377, 0.4438, 0.4728)
SVHN_std = (0.1980, 0.2010, 0.1970)


def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_stl10(args, root):

    transform_labeled = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    # base_dataset = datasets.STL10(
    #     root, split="train", download=True)

    train_labeled_dataset = datasets.STL10(root, split="train", transform=transform_labeled, download=True)
    train_unlabeled_dataset = datasets.STL10(root, split="unlabeled",
                                                         transform=TransformFixMatchSTL(mean=stl10_mean, std=stl10_std),
                                                         download=True)
    test_dataset = datasets.STL10(root, split="test", transform=transform_val, download=True)


    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    train_labeled_idxs = train_split_l(train_labeled_dataset.labels, l_samples, args)
    train_labeled_dataset = make_imbalance(train_labeled_dataset, train_labeled_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_svhn(args, root):
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(SVHN_mean, SVHN_std)
    # ])

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_mean, SVHN_std)
    ])


    base_dataset = datasets.SVHN(root, split='train', download=True)
    test_dataset = datasets.SVHN(root, split='test', download=True)

    test_idxs = testsplit(test_dataset.labels)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.labels, l_samples, u_samples, args)

    train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split='train', transform=transform_labeled)
    train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, train_labeled_idxs, split='train',
                                             transform=TransformFixMatch(mean=SVHN_mean, std=SVHN_std))
    test_dataset = SVHN_labeled(root, test_idxs, split='test', transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_smallimagenet(args, root):
    assert args.img_size == 32 or args.img_size == 64, 'img size should only be 32 or 64!!!'
    base_dataset = SmallImageNet(root, args.img_size, True)

    labeled_percent = 0.1

    dataset_mean = (0.48109809, 0.45747185, 0.40785507)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255

    dataset_std = (0.26040889, 0.2532126, 0.26820634)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255

    transform_train = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=int(args.img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    # select labeled data and construct labeled dataset
    num_classes = len(set(base_dataset.targets))
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in base_dataset.targets:
        num_data_per_cls[l] += 1

    num_labeled_data_per_cls = [int(np.around(n * labeled_percent)) for n in num_data_per_cls]
    print('total number of labeled data is ', sum(num_labeled_data_per_cls))
    # train_labeled_idxs = train_split(base_dataset.targets, num_labeled_data_per_cls, num_classes, args.seed)
    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_data_per_cls, args)

    train_labeled_dataset = SmallImageNet(root, args.img_size, True, transform=transform_train, indexs=train_labeled_idxs)
    train_unlabeled_dataset = SmallImageNet(root, args.img_size, True,
                                            transform=TransformFixMatch(mean=dataset_mean, std=dataset_std, img_size=args.img_size))
    test_dataset = SmallImageNet(root, args.img_size, False, transform=transform_val)

    arr = np.array(num_labeled_data_per_cls)
    tar_index = np.argsort(-arr)
    tar_index = tar_index.tolist()

    for idx in range(len(train_labeled_dataset.targets)):
        train_labeled_dataset.targets[idx] = tar_index.index(train_labeled_dataset.targets[idx])

    for idx in range(len(train_unlabeled_dataset.targets)):
        train_unlabeled_dataset.targets[idx] = tar_index.index(train_unlabeled_dataset.targets[idx])

    for idx in range(len(test_dataset.targets)):
        test_dataset.targets[idx] = tar_index.index(test_dataset.targets[idx])

    train_unlabeled_dataset.targets = np.array(train_unlabeled_dataset.targets)
    train_unlabeled_dataset.targets[train_labeled_idxs] = -2

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, args):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


def train_split_l(labels, n_labeled_per_class, args):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs


def testsplit(labels):
    labels = np.array(labels)
    test_idxs=[]
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:1500])
    np.random.shuffle(test_idxs)
    return test_idxs


def make_imbalance(dataset, indexs):
    dataset.data = dataset.data[indexs]
    dataset.labels = dataset.labels[indexs]
    return dataset


def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    print(class_num_list)
    return list(class_num_list)


class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)


class TransformFixMatchSTL(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets[exindexs] = -2
            self.targets = self.targets[indexs]
            # self.targets = np.array(self.targets)[indexs]


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets[exindexs] = -2
            self.targets = self.targets[indexs]
            # self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN_labeled(torchvision.datasets.SVHN):

    def __init__(self, root, indexs=None, exindexs=[], split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_labeled, self).__init__(root, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)
            self.labels[exindexs] = -2
            self.labels = self.labels[indexs]
            # self.labels = np.array(self.labels)[indexs]
        self.data = transpose(self.data)
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index
        return img, target


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, exindexs=[], split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_unlabeled, self).__init__(root, indexs, exindexs, split=split,
                                                transform=transform, target_transform=target_transform,
                                                download=download)


class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, transform=None, target_transform=None, indexs=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1
        self.targets = [i - 1 for i in self.targets]
        self.data = np.vstack(self.data).reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC  shape(-1, 32, 32, 3)

        # if len(exindexs) > 0:
        #     self.targets = np.array(self.targets)
        #     self.targets[exindexs] = -2
            # self.targets = self.targets[indexs]

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index
        return img, target


    def __len__(self):
        return len(self.data)


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'stl10': get_stl10,
                   'svhn': get_svhn,
                   'smallimagenet': get_smallimagenet}


