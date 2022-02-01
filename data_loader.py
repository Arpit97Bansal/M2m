import torch
import os
import numpy.random as nr
import numpy as np
import bisect
from PIL import Image

from torchvision import transforms
import errno
import torchvision
import shutil
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from scipy import io
import random

num_test_samples_cifar10 = [1000] * 10
num_test_samples_cifar100 = [100] * 100

DATA_ROOT = os.path.expanduser('~/data')


def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if True and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif True and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


def get_oversampled(dataset, num_sample_per_class, batch_size, TF_train, TF_test, dataset_type, fractions, create):

    if create:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)

        root = './root_' + dataset_type + '/'

        del_folder(root)
        create_folder(root)
        counts = []

        for i in range(10):
            lable_root = root + str(i) + '/'
            create_folder(lable_root)
            counts.append(0)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            counts[label] += 1

        for i in range(10):
            fractions[i] = counts[i] * fractions[i]
            counts[i] = 0

        print(fractions)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            counts[label] += 1
            if counts[label] <= fractions[label]:
                img.save(root + str(label) + '/' + str(idx) + '.png')

        ########## test
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True)

        root = './root_test_' + dataset_type + '/'

        del_folder(root)
        create_folder(root)

        for i in range(10):
            lable_root = root + str(i) + '/'
            create_folder(lable_root)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            img.save(root + str(label) + '/' + str(idx) + '.png')

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + dataset_type + '/'
    root_test = './root_test_' + dataset_type + '/'

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=BalancedBatchSampler(trainset), num_workers=1,
                                              drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, testloader, testloader



def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test, dataset_type, fractions, create):

    if create:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)

        root = './root_' + dataset_type + '/'

        del_folder(root)
        create_folder(root)
        counts = []

        for i in range(10):
            lable_root = root + str(i) + '/'
            create_folder(lable_root)
            counts.append(0)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            counts[label] += 1

        for i in range(10):
            fractions[i] = counts[i] * fractions[i]
            counts[i] = 0

        print(fractions)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            counts[label] += 1
            if counts[label] <= fractions[label]:
                img.save(root + str(label) + '/' + str(idx) + '.png')

        ########## test
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True)

        root = './root_test_' + dataset_type + '/'

        del_folder(root)
        create_folder(root)

        for i in range(10):
            lable_root = root + str(i) + '/'
            create_folder(lable_root)

        for idx in range(len(trainset)):
            img, label = trainset[idx]
            img.save(root + str(label) + '/' + str(idx) + '.png')




    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + dataset_type + '/'
    root_test = './root_test_' + dataset_type + '/'

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, testloader, testloader


def smote(data, targets, n_class, n_max):
    aug_data = []
    aug_label = []

    for k in range(1, n_class):
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))

        # Augmentation with SMOTE ( k-nearest )
        if smote:
            for i in range(class_len):
                for j in range(class_len):
                    class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
            sorted_idx = np.argsort(class_dist)

            for i in range(n_max - class_len):
                lam = nr.uniform(0, 1)
                row_idx = i % class_len
                col_idx = int((i - row_idx) / class_len) % (class_len - 1)
                new_data = np.round(
                    lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])

                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)

    return np.array(aug_data), np.array(aug_label)


def get_smote(dataset,  num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)

    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    class_max = max(num_sample_per_class)
    aug_data, aug_label = smote(train_cifar.data, train_cifar.targets, nb_classes, class_max)

    train_cifar.targets = np.concatenate((train_cifar.targets, aug_label), axis=0)
    train_cifar.data = np.concatenate((train_cifar.data, aug_data), axis=0)

    print("Augmented data num = {}".format(len(aug_label)))
    print(train_cifar.data.shape)

    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds