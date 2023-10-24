import argparse
import os
import time
import librosa
import torchaudio
import torchaudio.transforms as transforms
import math
import ast

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from torchaudio.transforms import SpecAugment
from torchlibrosa.augmentation import SpecAugmentation
from data_construction import data_construction
from asc_model import ASTModel as AUD_NET

parser = argparse.ArgumentParser(description='ADVANCE')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lrf', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--audio_length", type=int, default=1000, help="the dataset spectrogram std")
parser.add_argument("--dataset_mean", type=float, default=-5.7243857, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=3.6037567, help="the dataset spectrogram std")
args = parser.parse_args()

normalizer = np.load('./weights/audio_feature_normalizer.npy')
mu    = normalizer[0]
sigma = normalizer[1]

# audio_conf      = {'target_length': args.audio_length, 'freqm': 24, 'timem': 192 }
# audio_conf      = {'target_length': args.audio_length, 'freqm': 8, 'timem': 40 }
audio_conf      = {'target_length': args.audio_length, 'freqm': 0, 'timem': 0 }
val_audio_conf  = {'target_length': args.audio_length, 'freqm': 0,  'timem': 0   }
test_audio_conf = {'target_length': args.audio_length, 'freqm': 0,  'timem': 0   }

def audio_extract(wav_file,sr =16000):
    wav = librosa.load(wav_file, sr=sr)[0]
    spec = librosa.core.stft(wav,n_fft=4096,hop_length=400,win_length=1024,window='hann',center=True,pad_mode='constant')  # spec.size [2049,401]# 401个帧，
    mel = librosa.feature.melspectrogram(S=np.abs(spec),sr=sr,n_mels=64,fmax=8000)
    logmel = librosa.core.power_to_db(mel[:, :400])
    return logmel.T.astype('float32')

class CVSDataset(Dataset):
    def __init__(self, data_dir, data_sample, data_label, seed, audio_conf):

        self.data_dir = data_dir
        self.data_sample = data_sample
        self.data_label = data_label
        self.index_list = [i for i in range(len(self.data_label))]
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.index_list)

        self.audio_conf = audio_conf
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.spec_augmenter=SpecAugmentation(time_drop_width=64, time_stripes_num=2,freq_drop_width=8, freq_stripes_num=2)


    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):

        # sound_path = os.path.join(self.data_dir, 'sound_npy', self.data_sample[item] + '.npy')
        sound_path = os.path.join(self.data_dir, 'sound', self.data_sample[self.index_list[item]] + '.wav')
        waveform, sr = torchaudio.load(sound_path)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                  window_type='hanning', num_mel_bins=64, dither=0.0, frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        # fbank = fbank.unsqueeze(0)
        # fbank = freqm(fbank)
        # fbank = timem(fbank)
        # fbank = fbank.squeeze(0)
        # fbank = torch.transpose(fbank, 0, 1)
        fbank = (fbank + 4.26) / (4.57 * 2)

        # sound = np.load(sound_path).astype(np.float32)    # 【400,64】
        # sound_tensor = torch.from_numpy(sound).unsqueeze(0).unsqueeze(0)
        # augmented_spec_tensor = self.spec_augmenter(sound_tensor)
        # augmented_sound = augmented_spec_tensor.squeeze(0).squeeze(0).numpy()

        # snd = np.clip((np.exp(snd / 10) - LOW) / (HIGH - LOW), 0, 1)

        # sound = audio_extract(sound_path)   # [400,64]

        # sound = ((sound - mu) / sigma).astype('float32')    # 【400,64】

        # Apply audio augmentation
        # augmentations = Compose([
        #     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1),
        #     TimeStretch(min_rate=0.8, max_rate=1.2, p=1),
        #     PitchShift(min_semitones=-4, max_semitones=4, p=1)
        # ])
        # augmented_sound = augmentations(samples=sound, sample_rate=16000)

        scene_label = self.data_label[self.index_list[item]]
        return fbank, scene_label

class CVSDatasetNoAugmentation(Dataset):
    def __init__(self, data_dir, data_sample, data_label,seed):
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.data_label = data_label
        self.index_list = [i for i in range(len(self.data_label))]
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        # sound_path = os.path.join(self.data_dir, 'sound_npy', self.data_sample[item] + '.npy')
        sound_path = os.path.join(self.data_dir, 'sound_npy', self.data_sample[self.index_list[item]] + '.npy')
        sound = np.load(sound_path).astype(np.float32)
        scene_label = self.data_label[item]
        return sound, scene_label


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

(train_sample, train_label, val_sample, val_label, test_sample, test_label) = data_construction(r'D:\code\dataset\ADVANCE')

train_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', train_sample, train_label, seed=args.seed,audio_conf=audio_conf)
val_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', val_sample, val_label, seed=args.seed,audio_conf=val_audio_conf)
test_dataset = CVSDataset(r'D:\code\dataset\ADVANCE', test_sample, test_label, seed=args.seed,audio_conf=test_audio_conf)

train_loader= DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=False, num_workers=6)
val_loader  = DataLoader(dataset=val_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)


def train_epoch(model, optimizer, train_loader, criterion, epoch):
    model.train()
    total = 0
    sum_loss = 0.0
    correct = 0.0
    for batch_idx, data in enumerate(train_loader):
        aud, scene_label = data
        aud, scene_label = aud.type(torch.FloatTensor).to(device), scene_label.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        scene_output = model(aud)
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
        matrix = self.matrix / np.sum(self.matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)

        plt.xticks(range(self.num_classes), self.labels, rotation=90)
        plt.yticks(range(self.num_classes), self.labels)
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.5, 1])

        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = "{:.2f}".format(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if matrix[y, x] > thresh else "black")
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
                aud, scene_label = data
                aud, scene_label = aud.type(torch.FloatTensor).to(device), scene_label.type(torch.LongTensor).to(device)
                result = model(aud)
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
                f"val_loss:{val_loss},val_acc:{val_acc},val_precision:{precision},val_recall:{recall},val_fscore:{fscore}")
    if loader == test_loader:
        labels = ['airport', 'beach', 'bridge', 'farmland', 'forest', 'grassland', 'harbour', 'lake', 'orchard', 'residential', 'sparse shrub land', 'sports land', 'train station']
        confusion = ConfusionMatrix(num_classes=13, labels=labels)
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                aud, scene_label = data
                aud, scene_label = aud.type(torch.FloatTensor).to(device), scene_label.type(torch.LongTensor).to(device)
                result = model(aud)
                loss += criterion(result, scene_label).item()
                pred = result.argmax(dim=1, keepdim=False)
                correct += ((pred == scene_label).sum().cpu().numpy())
                total += scene_label.size(0)
                predict_labels = np.concatenate((predict_labels, pred.cpu().numpy()))
                ground_labels = np.concatenate((ground_labels, scene_label.cpu().numpy()))
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AUD_NET(label_dim=13,fstride=10,tstride=10,input_fdim=64,input_tdim=args.audio_length,imagenet_pretrain=True, audioset_pretrain=True).to(device)

def main():
    # optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=0.01)
    # optimizer = torch.optim.Adam(pg, args.lr, weight_decay=5e-3, betas=(0.95, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(1,args.epochs,1)), gamma=0.95)
    pg = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in pg) / 1e6))

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-3)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criteon = nn.CrossEntropyLoss()
    best_acc = 0.0
    training_loss = []
    training_acc = []
    valing_loss = []
    valing_acc = []

    for epoch in range(1,args.epochs):
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

            torch.save(model.state_dict(), './weights/ASC1_model.pth')

    print('Best val epoch:', best_epoch, 'best_acc:', best_acc)

    model_weight_path = "./weights/ASC1_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print('loaded best model!')
    test_loss, test_acc, test_precision, test_recall, test_fscore = evalute(model, test_loader, criteon, epoch)
    print('Test acc:', test_acc, 'Test precision:', test_precision, 'Test recall:', test_recall, 'Test fscore:', test_fscore)
    matplot_loss(training_loss, valing_loss)
    matplot_acc(training_acc, valing_acc)


if __name__ == '__main__':
    main()
