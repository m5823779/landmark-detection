import os, csv
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader


class monitor_dataset(data.Dataset):
    def __init__(self, data_path, train_list, transform):
        self.transform = transform
        self.data_dir = os.path.join(data_path)
        self.train_list = train_list
        self.imgs = []

        self.imgs = [self.data_dir + '/' + i for i in self.train_list]

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


DIR = "./monitor/train/"
files = os.listdir(DIR)

data_transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(256), transforms.ToTensor()])
train_list = []
for file in files:
        train_list.append(file)

dset = monitor_dataset(data_path=DIR, train_list=train_list, transform=data_transform)
batch_size = 10
num_workers = batch_size
loader = DataLoader(dset, batch_size=batch_size)
acc = np.zeros((3, 256, 256))
sq_acc = np.zeros((3, 256, 256))
for batch_idx, imgs in enumerate(loader):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs ** 2, axis=0)
    if batch_idx % 1 == 0:
        print('Accumulated {:d} / {:d}'.format(batch_idx * batch_size, len(dset)))

N = len(dset) * acc.shape[1] * acc.shape[2]
mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
mean_p /= N
print('Mean pixel = ', mean_p)
std_p = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std. pixel = ', std_p)

output_filename = os.path.join(DIR, '../../', 'stats.txt')
np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')