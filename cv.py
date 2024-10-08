import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model import *
from utils import Averager, ensure_path
from sklearn.model_selection import KFold
from eeg_dataset import eegDataset3, eegDataset2,eegDataset,eegDataset4
import pickle

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)num_epochs:" + str(args.max_epoch) +
                   "\n5)batch_size:" + str(args.batch_size) +
                   "\n6)input_shape:" + str(args.input_shape) + '\n')
        file.close()

    def load_per_subject(self, sub):
        save_path = os.getcwd()
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        #不同device的通道选择
        # channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]
        # data = data[:, :, np.array(channel_8), :]
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def n_fold_TVT(self, subject=[0], fold=5):
        def cv_split(k, k_fold, trial_num_sf, DATA, LABELS):
            if k < (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [(k + 1) * trial_num_sf + i for i in range(trial_num_sf)]
            elif k == (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [0 * trial_num_sf + i for i in range(trial_num_sf)]
            val_data = DATA[np.array(val_trial_choose), :, :]
            test_data = DATA[np.array(test_trial_choose), :, :]
            val_labels = LABELS[np.array(val_trial_choose)]
            test_labels = LABELS[np.array(test_trial_choose)]
            train_data = np.delete(DATA, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            train_labels = np.delete(LABELS, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            return train_data, train_labels, val_data, val_labels, test_data, test_labels
        #data are divided into train,validation,test set seperately,8:1:1
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        for sub in subject:
            data, label = self.load_per_subject(sub)
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            num_val_samples = data.shape[0] // fold
            for idx_fold in range(fold):
                train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                    cv_split(idx_fold, fold, num_val_samples, data, label)
                train_data = torch.from_numpy(train_data).float()
                train_labels = torch.from_numpy(train_labels).long()
                val_data = torch.from_numpy(val_data).float()
                val_labels = torch.from_numpy(val_labels).long()
                test_data = torch.from_numpy(test_data).float()
                test_labels = torch.from_numpy(test_labels).long()
                if self.args.reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act = test(args=self.args, data=test_data, label=test_labels,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0
                else:
                    # to train new models
                    acc_val, f1_val = train(args=self.args,
                                data_train=train_data,
                                label_train=train_labels,
                                data_val=val_data,
                                label_val=val_labels,
                                subject=subject,
                                fold=fold)

                    acc_test, pred, act = test(args=self.args, data=test_data, label=test_labels,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)
            #
            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            result = '{},{}'.format(tta[-1], f1)
            self.log2txt(result)

        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = 'test mAcc={} mF1={} val mAcc={} val F1={}'.format(mACC,
                                                                     mF1, mACC_val, mF1_val)
        self.log2txt(results)

    def n_fold_CV(self, subject=[0], fold=5, shuffle=False):

        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        save_path = os.getcwd()
        data_type = 'data_{}_{}'.format(self.args.data_format, 'KU')
        name = 'crosssubject2.hdf'
        path = osp.join(save_path, data_type, name)
        dataset = h5py.File(path, 'r')

        DATA = eegDataset(np.array(dataset['Source_Data_N']), np.array(dataset['label']))
        # DATA = eegDataset3(np.array(dataset['data']), np.array(dataset['data_4_8']), np.array(dataset['data_8_12']),
        #                    np.array(dataset['data_12_38']), np.array(dataset['label']))
        data_index = np.zeros((np.array(dataset['Source_Data_N']).shape[0],np.array(dataset['Source_Data_N']).shape[1]))

        va_val = Averager()
        vf_val = Averager()
        kf = KFold(n_splits=fold, shuffle=shuffle)
        for idx_fold, (idx_train, idx_val) in enumerate(kf.split(data_index)):
            if idx_fold == 1 :
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                #
                # train_dataloader = eegDataset2(DATA.data, DATA.data_4_8,DATA.data_8_12,
                #                                            DATA.data_12_38, DATA.label, idx_train)
                # val_dataloader = eegDataset2(DATA.data, DATA.data_4_8,DATA.data_8_12,
                #                                            DATA.data_12_38, DATA.label, idx_val)
                train_dataloader = eegDataset4(DATA.x, DATA.y, idx_train)
                val_dataloader = eegDataset4(DATA.x, DATA.y, idx_val)
                # acc_test, pred, act = test(args=self.args, test_loader=val_dataloader, fold=idx_fold)
                acc_val, f1_val = train(args=self.args, data_train=train_dataloader,
                                        data_val=val_dataloader, fold=idx_fold)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                tva.append(va_val.item())
                tvf.append(vf_val.item())
        tva = np.array(tva)
        tvf = np.array(tvf)

        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        #save_path
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        # results = 'test mAcc={} mF1={} val mAcc={} val F1={}'.format(mACC,
        # mF1, mACC_val, mF1_val)
        results = 'val mAcc={} val F1={}'.format(mACC_val, mF1_val)

        path = osp.join(save_path, 'save_cross/result/valbest_noteception.txt')
        np.savetxt(path, tva, fmt='%3f', delimiter=',')

        self.log2txt(results)

    def second_stage(self, fold=5, shuffle=False):

        save_path = os.getcwd()
        path = osp.join(save_path, 'save_cross/result/valbest.txt')
        val_teacher = np.loadtxt(path, delimiter=',')
        val_best = np.max(val_teacher)
        fold_best = np.argmax(val_teacher)
        # fold_best=5

        tva = []  # total validation accuracy
        tvf = []  # total validation f1

        data_type = 'data_{}_{}'.format(self.args.data_format, 'KU')
        name = 'crosssubject2.hdf'
        path = osp.join(save_path, data_type, name)
        dataset = h5py.File(path, 'r')
        channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]

        DATA_stu = eegDataset(np.array(dataset['Source_Data_N'])[:, :, np.array(channel_8), :],
                           np.array(dataset['label']))
        DATA_tea = eegDataset(np.array(dataset['Source_Data_N']), np.array(dataset['label']))
        data_index = np.zeros((np.array(dataset['Source_Data_N']).shape[0],np.array(dataset['Source_Data_N']).shape[1]))

        va_val = Averager()
        vf_val = Averager()

        kf = KFold(n_splits=fold, shuffle=shuffle)
        for idx_fold, (idx_train, idx_val) in enumerate(kf.split(data_index)):
            fold_best=1 #别忘注释掉
            val_best = 0.7921 #别忘注释掉
            if idx_fold == fold_best:
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                idx_train = np.arange(400)
                # train_dataloader = eegDataset2(DATA.data, DATA.data_4_8,DATA.data_8_12,
                #                                            DATA.data_12_38, DATA.label, idx_train)
                # val_dataloader = eegDataset2(DATA.data, DATA.data_4_8,DATA.data_8_12,
                #                                            DATA.data_12_38, DATA.label, idx_val)
                train_dataloader_stu = eegDataset4(DATA_stu.x, DATA_stu.y, idx_train)
                val_dataloader_stu = eegDataset4(DATA_stu.x, DATA_stu.y, idx_val)
                train_dataloader_tea = eegDataset4(DATA_tea.x, DATA_tea.y, idx_train)
                # val_dataloader_tea = eegDataset4(DATA_tea.x, DATA_tea.y, idx_val)
                del DATA_stu, DATA_tea
        # self.args.model = 'IGNNNet_stu'
        # self.args.Input_shape=(1, 8, 1000)
        acc_val, f1_val = combine_train(args=self.args,
                                        train_stu=train_dataloader_stu,
                                        val_stu=val_dataloader_stu,
                                        train_tea=train_dataloader_tea,
                                        fold=fold_best, val_best=val_best)

        va_val.add(acc_val)
        vf_val.add(f1_val)

        tva.append(va_val.item())
        tvf.append(vf_val.item())

        tva = np.array(tva)
        tvf = np.array(tvf)

        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = 'val mAcc={} val F1={}'.format(mACC_val, mF1_val)

        path = osp.join(save_path, 'save_cross/result/valfinal_noteception.txt')
        np.savetxt(path, tva, fmt='%3f', delimiter=',')

        self.log2txt(results)

    def check_lanse(self, subject=[0]):
        self.args.model = 'IGNNNet_stu'
        self.args.Input_shape = (1, 8, 1000)
        tta = []  # total test accuracy
        ttf = []  # total test f1
        preds, acts = [], []
        save_path = os.getcwd()
        for sub in subject:
            data_type = 'data_{}_{}'.format(self.args.data_format, 'lanse')
            sub_code = 'sub' + str(sub) + '.hdf'
            path = osp.join(save_path, data_type, sub_code)
            dataset = h5py.File(path, 'r')
            test_dataloader = eegDataset(np.array(dataset['data']), np.array(dataset['label']))

            # data_type = 'data_{}_{}'.format(self.args.data_format, 'KU')
            # name = 'crosssubject2.hdf'
            # path = osp.join(save_path, data_type, name)
            # dataset = h5py.File(path, 'r')
            # channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]
            #
            # DATA = eegDataset(np.array(dataset['Source_Data_N'])[:, :, np.array(channel_8), :],
            #                   np.array(dataset['label']))
            # idx_train = np.arange(400)
            # test_dataloader = eegDataset4(DATA.x, DATA.y, idx_train)


            acc_test, pred, act = test(args=self.args, test_loader=test_dataloader, fold=0)
            preds.extend(pred)
            acts.extend(act)
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc_test)
            ttf.append(f1)
        tta = np.array(tta)
        ttf = np.array(ttf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        path = osp.join(save_path, 'save_cross/result/无第二次训练.txt')
        np.savetxt(path, tta, fmt='%3f', delimiter=',')

    def check_lanse2(self, subject=[0]):
        def cv_split(k, k_fold, trial_num_sf, DATA, LABELS):
            if k < (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [(k + 1) * trial_num_sf + i for i in range(trial_num_sf)]
            elif k == (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [0 * trial_num_sf + i for i in range(trial_num_sf)]
            val_data = DATA[np.array(val_trial_choose), :, :]
            test_data = DATA[np.array(test_trial_choose), :, :]
            val_labels = LABELS[np.array(val_trial_choose)]
            test_labels = LABELS[np.array(test_trial_choose)]
            train_data = np.delete(DATA, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            train_labels = np.delete(LABELS, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            return train_data, train_labels, val_data, val_labels, test_data, test_labels

        # self.args.model = 'EEGNet8_2'
        self.args.model = 'IGNNNet_stu'
        self.args.Input_shape = (1, 8, 1000)
        self.args.max_epoch=800
        self.args.patient=100
        self.args.batch_size=4
        self.args.learning_rate=1e-4
        sub_acc = []  # total test accuracy
        ttf = []  # total test f1
        acc_test, preds, acts = [], [], []
        save_path = os.getcwd()
        for sub_fold, sub in enumerate(subject):
            data_type = 'data_{}_{}'.format(self.args.data_format, 'lanse')
            sub_code = 'sub' + str(sub) + '.hdf'
            path = osp.join(save_path, data_type, sub_code)
            dataset = h5py.File(path, 'r')
            num_val_samples = 80//5
            for idx_fold in range(5):
                train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                    cv_split(idx_fold, 5, num_val_samples, np.array(dataset['data']), np.array(dataset['label']))
                # train_data_4_8, train_labels, val_data_4_8, val_labels, test_data_4_8, test_labels = \
                #     cv_split(idx_fold, 5, num_val_samples, np.array(dataset['data_4_8']), np.array(dataset['label']))
                # train_data_8_12, train_labels, val_data_8_12, val_labels, test_data_8_12, test_labels = \
                #     cv_split(idx_fold, 5, num_val_samples, np.array(dataset['data_8_12']), np.array(dataset['label']))
                # train_data_12_38, train_labels, val_data_12_38, val_labels, test_data_12_38, test_labels = \
                #     cv_split(idx_fold, 5, num_val_samples, np.array(dataset['data_12_38']), np.array(dataset['label']))

                train_dataloader = eegDataset(train_data, train_labels)
                val_dataloader = eegDataset(val_data, val_labels)
                test_dataloader = eegDataset(test_data, test_labels)

                # train_dataloader = eegDataset3(train_data, train_data_4_8, train_data_8_12, train_data_12_38, train_labels)
                # val_dataloader = eegDataset3(val_data, val_data_4_8, val_data_8_12, val_data_12_38, val_labels)
                # test_dataloader = eegDataset3(test_data, test_data_4_8, test_data_8_12, test_data_12_38, test_labels)
                acc_val, f1_val = train(args=self.args, data_train=train_dataloader, data_val=val_dataloader,
                                        fold=idx_fold)
                acc_test, pred, act = test(args=self.args, test_loader=test_dataloader, fold=idx_fold)
                if idx_fold == 0:
                    acts = [acc_test.tolist()]
                else:
                    acts.append(acc_test.tolist())
                # preds.extend(pred)
                # acts.extend(act)
                # acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
                # tta.append(acc)
                # ttf.append(f1)
            tta = np.array(acts)
            # ttf = np.array(ttf)
            mACC = np.mean(tta)
            # mF1 = np.mean(ttf)
            std = np.std(tta)
            if sub_fold == 0:
                sub_acc = [mACC.tolist()]
            else:
                sub_acc.append(mACC.tolist())
            path = osp.join(save_path, 'save_cross/result/testfinal.txt')
            np.savetxt(path, np.array(sub_acc), fmt='%3f', delimiter=',')
            print('Final: test mean ACC:{} std:{}'.format(mACC, std))

    def check_lanse3(self, subject=[0]):

        # self.args.model = 'EEGNet8_2'
        self.args.model = 'IGNNNet_stu'
        self.args.Input_shape = (1, 8, 1000)
        self.args.max_epoch = 800
        self.args.patient = 100
        self.args.batch_size = 4
        self.args.learning_rate = 1e-4
        sub_acc = []  # total test accuracy
        ttf = []  # total test f1
        acc_test, preds, acts = [], [], []
        save_path = os.getcwd()
        for sub_fold, sub in enumerate(subject):
            data_type = 'data_{}_{}'.format(self.args.data_format, 'lanse')
            sub_code = 'sub' + str(sub) + '.hdf'
            path = osp.join(save_path, data_type, sub_code)
            dataset = h5py.File(path, 'r')
            val_data, val_labels = np.array(dataset['data'])[40:60],np.array(dataset['label'])[40:60]
            test_data, test_labels = np.array(dataset['data'])[60:],np.array(dataset['label'])[60:]
            stride = 80//10
            for idx_fold in range(5):
                if idx_fold == 4:
                    train_data, train_labels = np.array(dataset['data'])[0:stride * (idx_fold + 1)], np.array(
                        dataset['label'])[0:stride * (idx_fold + 1)]

                    train_dataloader = eegDataset(train_data, train_labels)
                    val_dataloader = eegDataset(val_data, val_labels)
                    test_dataloader = eegDataset(test_data, test_labels)

                    acc_val, f1_val = train(args=self.args, data_train=train_dataloader, data_val=val_dataloader,
                                            fold=idx_fold)
                    acc_test, pred, act = test(args=self.args, test_loader=test_dataloader, fold=idx_fold)
                    if idx_fold == 0:
                        acts = [acc_test.tolist()]
                    else:
                        acts.append(acc_test.tolist())
                    # preds.extend(pred)
                    # acts.extend(act)
                    # acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
                    # tta.append(acc)
                    # ttf.append(f1)
            tta = np.array(acts)
            # ttf = np.array(ttf)
            mACC = np.mean(tta)
            # mF1 = np.mean(ttf)
            std = np.std(tta)
            if sub_fold == 0:
                sub_acc = [mACC.tolist()]
            else:
                sub_acc.append(mACC.tolist())
        path = osp.join(save_path, 'save_cross/result/testfinal.txt')
        np.savetxt(path, np.array(sub_acc), fmt='%3f', delimiter=',')
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))

    def check_lanse4(self, subject=[0]):
        def cv_split(k, k_fold, trial_num_sf, DATA, LABELS):
            if k < (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [(k + 1) * trial_num_sf + i for i in range(trial_num_sf)]
            elif k == (k_fold - 1):
                val_trial_choose = [k * trial_num_sf + i for i in range(trial_num_sf)]
                test_trial_choose = [0 * trial_num_sf + i for i in range(trial_num_sf)]
            val_data = DATA[np.array(val_trial_choose), :, :]
            test_data = DATA[np.array(test_trial_choose), :, :]
            val_labels = LABELS[np.array(val_trial_choose)]
            test_labels = LABELS[np.array(test_trial_choose)]
            train_data = np.delete(DATA, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            train_labels = np.delete(LABELS, np.concatenate([val_trial_choose, test_trial_choose], axis=0), axis=0)
            return train_data, train_labels, val_data, val_labels, test_data, test_labels

        # self.args.model = 'EEGNet8_2'
        self.args.model = 'IGNNNet_stu'
        self.args.Input_shape = (1, 8, 1000)
        self.args.max_epoch=800
        self.args.patient=100
        self.args.batch_size=4
        self.args.learning_rate=1e-4
        sub_acc = []  # total test accuracy
        ttf = []  # total test f1
        acc_test, preds, acts = [], [], []
        save_path = os.getcwd()
        for sub_fold, sub in enumerate(subject):
            data_type = 'data_{}_{}'.format(self.args.data_format, 'lanse')
            sub_code = 'sub' + str(sub) + '.hdf'
            path = osp.join(save_path, data_type, sub_code)
            dataset = h5py.File(path, 'r')
            num_val_samples = 80//5
            for idx_fold in range(5):
                train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                    cv_split(idx_fold, 5, num_val_samples, np.array(dataset['data']), np.array(dataset['label']))

                train_dataloader = eegDataset(train_data, train_labels)
                val_dataloader = eegDataset(val_data, val_labels)
                test_dataloader = eegDataset(test_data, test_labels)

                acc_val, f1_val = train(args=self.args, data_train=train_dataloader, data_val=val_dataloader,
                                        fold=idx_fold)
                acc_test, pred, act = test(args=self.args, test_loader=test_dataloader, fold=idx_fold)
                if idx_fold == 0:
                    acts = [acc_test.tolist()]
                else:
                    acts.append(acc_test.tolist())

            tta = np.array(acts)
            mACC = np.mean(tta)
            std = np.std(tta)
            if sub_fold == 0:
                sub_acc = [mACC.tolist()]
            else:
                sub_acc.append(mACC.tolist())
        path = osp.join(save_path, 'save_cross/result/testfinal.txt')
        np.savetxt(path, np.array(sub_acc), fmt='%3f', delimiter=',')
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))

    def log2txt(self, content):
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()

