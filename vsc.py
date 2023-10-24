import argparse
import os
import time
import math

from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support


import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from VSC_model import VSC_Net
from resnet_image import resnet101
from data_construction import data_construction
from vsc_model import vit_base_patch16_224_in21k as create_model



def augment_image(image):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image


class CVSDataset(Dataset):
    def __init__(self, data_dir, data_sample, data_label, seed, enhance=False, transform=None):  # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.data_label = data_label
        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.index_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        # return (img_data,audio_data,label)

        image_path = os.path.join(self.data_dir, 'vision', self.data_sample[self.index_list[item]] + '.jpg')
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.enable_enhancement:
            image = augment_image(image)

        scene_label = self.data_label[self.index_list[item]]

        return image, scene_label


parser = argparse.ArgumentParser(description='ADVANCE')
parser.add_argument('--freeze_layers', type=bool, default=False)
parser.add_argument('--weights', type=str, default='./weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                    help='initial weights path')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lrf', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=10)
args = parser.parse_args()


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

(train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_construction(
    r'D:\code\dataset\ADVANCE')

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

train_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', train_sample, train_label, seed=args.seed,transform=data_transform["train"])
val_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', val_sample, val_label, seed=args.seed,transform=data_transform["val"])
test_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', test_sample, test_label, seed=args.seed,transform=data_transform["test"])

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,pin_memory=True, shuffle=True, num_workers=6)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,pin_memory=True, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,pin_memory=True, shuffle=False, num_workers=6)


def train_epoch(model, optimizer, train_loader, criterion, epoch):
    model.train()
    total = 0
    sum_loss = 0.0
    correct = 0.0
    for batch_idx, data in enumerate(train_loader):
        img, scene_label = data
        img, scene_label = img.to(device), scene_label.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        scene_output = model(img)
        loss = criterion(scene_output, scene_label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        total += scene_label.size(0)
        predict_label = torch.argmax(scene_output, dim=1)
        correct += ((predict_label == scene_label).sum().cpu().numpy())

    train_acc = correct / total
    train_loss = sum_loss / (batch_idx + 1)

    print(f"train,epoch{epoch},loss:{train_loss},acc:{correct / total}")
    return train_loss, train_acc


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        sum_F1Score = 0.0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model ConfusionMatrix accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1Score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1Score = round((2 * Precision * Recall) / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            sum_F1Score += F1Score

            table.add_row([self.labels[i], Precision, Recall, Specificity, F1Score])
        print(table)
        print(f"ConfusionMatrix test_F1Score:{sum_F1Score / self.num_classes}")

    def plot(self):
        matrix = self.matrix
        # 计算混淆矩阵的归一化值
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums

        plt.imshow(normalized_matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示color-bar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 5
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = "{:.2f}".format(normalized_matrix[y, x]) if normalized_matrix[y, x] != 0 else "0"  # 将非零值显示为整数，零值显示为字符串"0"
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if normalized_matrix[y, x] > thresh else "black")
        plt.tight_layout()
        plt.show()


def evalute(model, loader, criterion, epoch):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    ground_labels = np.array([])
    predict_labels = np.array([])
    if loader == val_loader:
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                img, scene_label = data
                img, scene_label = img.type(torch.FloatTensor).to(device), scene_label.type(torch.LongTensor).to(device)
                result = model(img)
                loss += criterion(result, scene_label).item()
                pred = result.argmax(dim=1, keepdim=False)
                correct += ((pred == scene_label).sum().cpu().numpy())
                total += scene_label.size(0)
                predict_labels = np.concatenate((predict_labels, pred.cpu().numpy()))
                ground_labels = np.concatenate((ground_labels, scene_label.cpu().numpy()))

            val_loss = loss / (batch_idx + 1)
            val_acc = correct / total
            (precision, recall, fscore, sup) = sklearn.metrics.precision_recall_fscore_support(ground_labels,
                                                                                               predict_labels,
                                                                                               average='weighted',
                                                                                               zero_division=1)
            print(
                f"val_epoch{epoch},val_loss:{val_loss},val_acc:{val_acc},val_precision:{precision},val_recall:{recall},val_fscore:{fscore}")
    if loader == test_loader:
        labels = ['airport', 'beach', 'bridge', 'farmland', 'forest', 'grassland', 'harbour', 'lake', 'orchard', 'residential', 'sparse shrub land', 'sports land', 'train station']
        confusion = ConfusionMatrix(num_classes=13, labels=labels)
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                img, scene_label = data
                img, scene_label = img.type(torch.FloatTensor).to(device), scene_label.type(torch.LongTensor).to(device)
                result = model(img)
                loss += criterion(result, scene_label).item()
                pred = result.argmax(dim=1, keepdim=False)
                correct += ((pred == scene_label).sum().cpu().numpy())
                total += scene_label.size(0)
                predict_labels = np.concatenate((predict_labels, pred.cpu().numpy()))
                ground_labels = np.concatenate((ground_labels, scene_label.cpu().numpy()))

                result = torch.softmax(result, dim=1)
                result = torch.argmax(result, dim=1)
                confusion.update(result.to("cpu").numpy(), scene_label.to("cpu").numpy())

            val_loss = loss / (batch_idx + 1)
            val_acc = correct / total
            (precision, recall, fscore, sup) = sklearn.metrics.precision_recall_fscore_support(ground_labels,
                                                                                               predict_labels,
                                                                                               average='weighted',
                                                                                               zero_division=1)
            print(
                f"test_loss:{val_loss},test_acc:{val_acc},test_precision:{precision},test_recall:{recall},test_fscore:{fscore}")
            confusion.summary()
            confusion.plot()

    return val_loss, val_acc, precision, recall, fscore


def matplot_loss(train_loss, val_loss):
    minmum_train_index = np.argmin(train_loss)
    minmum_train = np.min(train_loss)
    minmum_val_index = np.argmin(val_loss)
    minmum_val = np.min(val_loss)

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.plot(minmum_val_index, minmum_val, 'r*')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='best')
    plt.show()
    plt.close()


def matplot_acc(train_acc, val_acc):
    maxmum_val_index = np.argmax(val_acc)
    maxmum_val = np.max(val_acc)

    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.plot(maxmum_val_index, maxmum_val, 'r*')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("train_dataset and val_dataset acc")
    plt.show()
    plt.close()


device = torch.device('cuda')

model = create_model(num_classes=13, has_logits=False).to(device)

if args.weights != "":
    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    weights_dict = torch.load(args.weights, map_location=device)
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    model.load_state_dict(weights_dict, strict=False)

def main():

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    criteon = nn.CrossEntropyLoss().to(device)
    best_acc = 0.0
    training_loss = []
    training_acc = []
    valing_loss = []
    valing_acc = []

    for epoch in range(1, args.epochs):

        start_train = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, criteon, epoch)
        val_loss, val_acc, val_precision, val_recall, val_fscore = evalute(model, val_loader, criteon, epoch)
        scheduler.step()
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        valing_loss.append(val_loss)
        valing_acc.append(val_acc)
        end_train = time.time()
        print(f"Epoch_{epoch}_cost time: {end_train - start_train}s")

        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/VSC_VIT_model.pth")

    print('Best val epoch:', best_epoch, 'best_acc:', best_acc)

    model_weight_path = "./weights/VSC_VIT_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    print('loaded best model!')

    test_loss, test_acc, test_precision, test_recall, test_fscore = evalute(model, test_loader, criteon,
                                                                            epoch)
    print('Test acc:', test_acc, 'Test precision:', test_precision, 'Test recall:', test_recall, 'Test fscore:',
          test_fscore)
    matplot_loss(training_loss, valing_loss)
    matplot_acc(training_acc, valing_acc)


if __name__ == '__main__':
    main()
