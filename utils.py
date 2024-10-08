import os
import time
import h5py
import numpy as np
import pprint
import random

from networks import *
from Proposed import IGNNNet, IGNNNet_stu
from public_model import *
from eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from EEGGENET import EEGGENET
from EEGARNN import EEGARNN
from EEGConformer import EEGConformerModule
from ATCNet import ATCNetModule

def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    if args.model == 'LGGNet':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = LGGNet(
            num_classes=args.num_class, input_size=input_size,
            sampling_rate=int(args.sampling_rate*args.scale_coefficient),
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    if args.model == 'ShallowConvNet':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = ShallowConvNet(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'IGNNNet':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = IGNNNet(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'EEGNet':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = EEGNet8_2(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'DeepConvNet':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = DeepConvNet(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'TDenseNet':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = TDenseNet(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'IGNNNet_stu':
        input_size = (args.input_shape[0], args.input_shape[1], args.input_shape[2])
        model = IGNNNet_stu(
            num_classes=args.num_class, input_size=input_size)
    if args.model == 'EEGGENET':
        '''
        Three types of Adj
        '''
        # Adj = torch.tensor(load_adj('bciciv2a'), dtype=torch.float32)
        Adj = torch.eye(args.input_shape[1])

        model = EEGGENET(Adj, args.input_shape[1], 2, k=1, input_time_length=args.input_shape[2], Adj_learn=True,
                                    drop_prob=0.2, pool_mode='mean', f1=8, f2=16, kernel_length=64)
    if args.model == 'EEGARNN':
        Abf = np.ones((args.input_shape[1], args.input_shape[1]))
        np.fill_diagonal(Abf, 0)
        A = preprocess_adj(Abf)
        A = np.float32(A)
        A = torch.from_numpy(A)
        model = EEGARNN((args.input_shape[1], args.input_shape[2]), A)
    if args.model == 'EEGConformer':
        model = EEGConformerModule(in_channels=args.input_shape[1], embedding_size=40, depth=6, n_classes=2,
                                   input_size_cls=2440)
    if args.model == 'ATCNet':
        model = ATCNetModule(in_channels=args.input_shape[1], n_classes=2)

    return model


def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err

def normalize_adj(adj):
    d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def preprocess_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj)
    return adj

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class EarlyStopping:
    def __init__(self, patience=100, delta=0, checkpoint_path='checkpoint.pt',best_score=None):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = best_score   # 用于two-stage，如果max或min达到阈值，则立即停止训练
        self.early_stop = False
        self.value = np.Inf
        self.f1score = None
        self.best_acc = None
        self.best_loss = None

    def __call__(self, value, model, mode = None, f1score=None):
        if mode == 'loss':
            if self.best_loss is None:
                self.best_loss = value
                self.save_checkpoint(value, model, f1score)
            elif value > self.best_loss + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('early stopping')
            elif self.best_score is not None:
                if value < self.best_score:
                    self.save_checkpoint(value, model, f1score)
                    self.early_stop = True
                    print('early stopping')
            else:
                self.best_loss = value
                self.save_checkpoint(value, model, f1score)
                self.counter = 0
        elif mode == 'acc':
            if self.best_acc is None:
                self.best_acc = value
                self.save_checkpoint(value, model, f1score)
            elif value < self.best_acc + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('early stopping')
            elif self.best_score is not None:
                if value > self.best_score:
                    self.save_checkpoint(value, model, f1score)
                    self.early_stop = True
                    print('early stopping')
            else:
                self.best_acc = value
                self.save_checkpoint(value, model, f1score)
                self.counter = 0

    def save_checkpoint(self, value, model, f1score=None):
        # torch.save(model.state_dict(), self.checkpoint_path)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.value = value
        self.f1score = f1score

def load_adj(dn='bciciv2a', norm=False):
    if 'hgd' == dn:
        num_node = 44
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 21), (1, 11), (1, 25), (1, 14),
                         (2, 22), (2, 37), (2, 11), (2, 12), (2, 26), (2, 15), (2, 39),
                         (3, 38), (3, 23), (3, 12), (3, 13), (3, 40), (3, 16), (3, 27),
                         (4, 24), (4, 13), (4, 28), (4, 17),
                         (5, 25), (5, 11), (5, 26), (5, 14), (5, 15), (5, 29), (5, 18), (5, 30),
                         (6, 27), (6, 13), (6, 28), (6, 16), (6, 17), (6, 31), (6, 20), (6, 32),
                         (7, 14), (7, 29), (7, 18), (7, 33),
                         (8, 30), (8, 15), (8, 41), (8, 18), (8, 19), (8, 34), (8, 43),
                         (9, 42), (9, 16), (9, 31), (9, 19), (9, 20), (9, 44), (9, 35),
                         (10, 17), (10, 32), (10, 20), (10, 36),
                         (11, 21), (11, 22), (11, 25), (11, 26),
                         (12, 37), (12, 38), (12, 39), (12, 40),
                         (13, 23), (13, 24), (13, 27), (13, 28),
                         (14, 25), (14, 29),
                         (15, 26), (15, 39), (15, 30), (15, 41),
                         (16, 40), (16, 27), (16, 42), (16, 31),
                         (17, 28), (17, 32),
                         (18, 29), (18, 30), (18, 33), (18, 34),
                         (19, 41), (19, 42), (19, 43), (19, 44),
                         (20, 31), (20, 32), (20, 35), (20, 36),
                         (21, 22), (21, 25),
                         (22, 37), (22, 26),
                         (23, 38), (23, 24), (23, 27),
                         (24, 28),
                         (25, 26), (25, 29),
                         (26, 39), (26, 30),
                         (27, 40), (27, 28), (27, 31),
                         (28, 32),
                         (29, 30), (29, 33),
                         (30, 41), (30, 34),
                         (31, 42), (31, 32), (31, 35),
                         (32, 36),
                         (33, 34),
                         (34, 43),
                         (35, 36), (35, 44),
                         (37, 38), (37, 39),
                         (38, 40),
                         (39, 40), (39, 41),
                         (40, 42),
                         (41, 43), (41, 42),
                         (42, 44),
                         (43, 44)]
    elif 'bciciv2a' == dn:
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 3), (1, 4), (1, 5),
                         (2, 3), (2, 7), (2, 8), (2, 9),
                         (3, 4), (3, 8), (3, 9), (3, 10),
                         (4, 5), (4, 9), (4, 10), (4, 11),
                         (5, 6), (5, 10), (5, 11), (5, 12),
                         (6, 11), (6, 12), (6, 13),
                         (7, 8), (7, 14),
                         (8, 9), (8, 14), (8, 15),
                         (9, 10), (9, 14), (9, 15), (9, 16),
                         (10, 11), (10, 15), (10, 16), (10, 17),
                         (11, 12), (11, 16), (11, 17), (11, 18),
                         (12, 13), (12, 17), (12, 18),
                         (13, 18),
                         (14, 15), (14, 19),
                         (15, 16), (15, 19), (15, 20),
                         (16, 17), (16, 19), (16, 20), (16, 21),
                         (17, 18), (17, 20), (17, 21),
                         (18, 21),
                         (19, 20), (19, 22),
                         (20, 21), (20, 22),
                         (21, 22)]
    else:
        raise ValueError('cant support {} dataset'.format(dn))
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
    edge = self_link + neighbor_link
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1.
        A[j, i] = 1.
    if norm:
        A = normalize_adj(torch.tensor(A, dtype=torch.float32), mode='sym')
    return A
