import numpy as np
import datetime
import os
import csv
import h5py
import os.path as osp
from train_model import *
from utils import Averager, ensure_path
from eeg_dataset import eegDataset,eegDataset4
from sklearn.metrics import cohen_kappa_score

ROOT = os.getcwd()


class two_stage:
    def __init__(self, args):
        self.args = args
        # Log the results per subject
        self.result_path = osp.join(args.save_path, str(args.model))
        ensure_path(self.result_path)
        data_type = 'data_{}_{}'.format(self.args.data_format, str(self.args.dataset))
        self.data_path = osp.join(os.getcwd(), data_type)

    def run(self,sub_to_run,data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:
            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

            #First stage:
            '''
            Training Session is divided into training set(75% volume) and validation set(25% volume)
            Using Early stopping monitor the validation loss
            '''
            split_index = int(len(Tsession_data) * 0.75)
            T_data = Tsession_data[:split_index]
            T_labels = Tsession_label[:split_index]
            V_data = Tsession_data[split_index:]
            V_labels = Tsession_label[split_index:]
            train_dataloader = eegDataset(T_data, T_labels)
            val_dataloader = eegDataset(V_data, V_labels)
            # acc_val, f1_val , model, pred_val, act_val= train(args=self.args, data_train=train_dataloader,
            #                         data_val=val_dataloader, sub=sub)
            # # Second stage:
            # '''
            # All training Session data will be used in trainloader and previous validation set(25% volume) is used for
            # mointering
            # Using Early stopping monitor the validation acc
            # '''
            # train_dataloader = eegDataset(Tsession_data, Tsession_label)
            # acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
            #                                                   data_val=val_dataloader, sub=sub,val_best=acc_val,
            #                                                   model_F = model)

            test_dataloader = eegDataset(Esession_data, Esession_label)
            model_path = self.result_path +'//sub' + str(sub) + '.pth'
            acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                              model_path = model_path)
            ACC_vals.append(acc_val)
            F1_vals.append(f1_val)
            ALL_pred_vals.extend(pred_val)
            ALL_act_vals.extend(act_val)

        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = self.result_path + '//results.txt'
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

        print("Finished!")

    def KU_transfer_TargetDataset_no_finetuning(self,sub_to_run,data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:
            model_path = " "
            result_path = " "
            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

                # Only 21 Channel transfer available
                channel_to_remove = 3  # FCz Channel cannot be transferred
                Esession_data = np.delete(Esession_data, channel_to_remove, axis=2)

            #No fine-tuning traing:
            test_dataloader = eegDataset(Esession_data, Esession_label)
            model_path = model_path + self.args.model + '.pth'
            acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                      model_path=model_path)
            ACC_vals.append(acc_val)
            F1_vals.append(f1_val)
            ALL_pred_vals.extend(pred_val)
            ALL_act_vals.extend(act_val)

        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = result_path + self.args.model  + '_results.txt'
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

    def KU_transfer_TargetDataset_finetuning(self, sub_to_run, data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:

            model_path = " "
            result_path = " "
            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

                # Only 21 Channel transfer available
                channel_to_remove = 3  # FCz Channel cannot be transferred
                Tsession_data = np.delete(Tsession_data, channel_to_remove, axis=2)
                Esession_data = np.delete(Esession_data, channel_to_remove, axis=2)
            # First stage:
            '''
            Training Session is divided into training set(75% volume) and validation set(25% volume)
            Using Early stopping monitor the validation loss
            '''
            split_index = int(len(Tsession_data) * 0.75)
            T_data = Tsession_data[:split_index]
            T_labels = Tsession_label[:split_index]
            V_data = Tsession_data[split_index:]
            V_labels = Tsession_label[split_index:]
            train_dataloader = eegDataset(T_data, T_labels)
            val_dataloader = eegDataset(V_data, V_labels)

            model = get_model(self.args)
            if CUDA:
                model = model.cuda()
            model_path = model_path + self.args.model + '.pth'
            model.load_state_dict(torch.load(model_path))
            self.args.save_path = result_path
            ensure_path(self.args.save_path)
            ensure_path(osp.join(self.args.save_path,str(self.args.model)))
            acc_val, f1_val, model, pred_val, act_val= train(args=self.args, data_train=train_dataloader,
                                    data_val=val_dataloader, sub=sub, model_F = model)
            # Second stage:
            '''
            All training Session data will be used in trainloader and previous validation set(25% volume) is used for
            mointering
            Using Early stopping monitor the validation acc
            '''
            train_dataloader = eegDataset(Tsession_data, Tsession_label)
            acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
                                                              data_val=val_dataloader, sub=sub,val_best=acc_val,
                                                              model_F = model)

            test_dataloader = eegDataset(Esession_data, Esession_label)
            model_path = osp.join(self.args.save_path,str(self.args.model),'{}.pth'.format('sub'+str(sub)))
            acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                              model_path = model_path)
            ACC_vals.append(acc_val)
            F1_vals.append(f1_val)
            ALL_pred_vals.extend(pred_val)
            ALL_act_vals.extend(act_val)

        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = osp.join(self.args.save_path,str(self.args.model),'result.txt')
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

        print("Finished!")

    def KU_transfer_TargetDataset_stu_train(self, sub_to_run, data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:

            model_path = " "
            result_path = " "
            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

                # Only 21 Channel transfer available
                channel_to_remove = 3  # FCz Channel cannot be transferred
                Tsession_data = np.delete(Tsession_data, channel_to_remove, axis=2)
                Esession_data = np.delete(Esession_data, channel_to_remove, axis=2)
            # First stage:
            '''
            Training Session is divided into training set(75% volume) and validation set(25% volume)
            Using Early stopping monitor the validation loss
            '''

            split_index = int(len(Tsession_data) * 0.75)
            T_data = Tsession_data[:split_index]
            T_labels = Tsession_label[:split_index]
            V_data = Tsession_data[split_index:]
            V_labels = Tsession_label[split_index:]
            train_dataloader = eegDataset(T_data, T_labels)
            val_dataloader = eegDataset(V_data, V_labels)

            model = get_model(self.args)
            if CUDA:
                model = model.cuda()
            model.load_state_dict(torch.load(model_path))
            self.args.save_path = result_path
            acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
                                                              data_val=val_dataloader, sub=sub, model_F=model)
            # Second stage:
            '''
            All training Session data will be used in trainloader and previous validation set(25% volume) is used for
            mointering
            Using Early stopping monitor the validation acc
            '''
            train_dataloader = eegDataset(Tsession_data, Tsession_label)
            acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
                                                              data_val=val_dataloader, sub=sub, val_best=None,
                                                              model_F=model)

            test_dataloader = eegDataset(Esession_data, Esession_label)
            model_path = osp.join(self.args.save_path, str(self.args.model), '{}.pth'.format('sub' + str(sub)))
            acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                      model_path=model_path)
            ACC_vals.append(acc_val)
            F1_vals.append(f1_val)
            ALL_pred_vals.extend(pred_val)
            ALL_act_vals.extend(act_val)

        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = osp.join(self.args.save_path, str(self.args.model), 'result.txt')
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

        print("Finished!")

    def KU_transfer_TargetDataset_stu_train_kfold(self, sub_to_run, data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:

            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

                # Only 21 Channel transfer available
                channel_to_remove = 3  # FCz Channel cannot be transferred
                Tsession_data = np.delete(Tsession_data, channel_to_remove, axis=2)
                Esession_data = np.delete(Esession_data, channel_to_remove, axis=2)

            Tsession_data = np.concatenate((Tsession_data, Esession_data), axis=0)
            Tsession_label = np.concatenate((Tsession_label, Esession_label), axis=0)
            kfold_ACC_vals = []
            kfold_F1_vals = []
            for kfold in range(5):

                num_val_samples = 80//5   # if dataset is BlueBCI, it should be 288//5
                train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                    self.cv_split(kfold, 5, num_val_samples, Tsession_data, Tsession_label)

                train_dataloader = eegDataset(train_data, train_labels)
                val_dataloader = eegDataset(val_data, val_labels)
                test_dataloader = eegDataset(test_data, test_labels)
                model = get_model(self.args)
                if CUDA:
                    model = model.cuda()

                model_path = " "
                result_path = " "

                model.load_state_dict(torch.load(model_path))
                self.args.save_path = result_path
                acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
                                                                  data_val=val_dataloader, sub=sub, model_F=model,
                                                                  kfold=kfold)
                model_path = osp.join(self.args.save_path, str(self.args.model))+'/sub'+str(sub)+'_'+str(kfold)+'.pth'
                acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                          model_path=model_path)
                kfold_ACC_vals.append(acc_val)
                kfold_F1_vals.append(f1_val)
                ALL_pred_vals.extend(pred_val)
                ALL_act_vals.extend(act_val)

            ACC_vals.append(np.mean(kfold_ACC_vals))
            F1_vals.append(np.mean(kfold_F1_vals))
        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = osp.join(self.args.save_path, str(self.args.model), 'result.txt')
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

        print("Finished!")
    def run_kfold(self, sub_to_run, data_type="data"):
        ACC_vals = []
        F1_vals = []
        ALL_pred_vals = []
        ALL_act_vals = []
        for sub in sub_to_run:
            sub_code = 'sub' + str(sub) + '.hdf'
            Tsession_path = osp.join(self.data_path, 'T', sub_code)
            dataset = h5py.File(Tsession_path, 'r')
            Tsession_data = np.array(dataset[data_type])
            Tsession_label = np.array(dataset['label'])
            Esession_path = osp.join(self.data_path, 'E', sub_code)
            dataset = h5py.File(Esession_path, 'r')
            Esession_data = np.array(dataset[data_type])
            Esession_label = np.array(dataset['label'])

            '''
               BCI_IV_2a has 4 tasks, while we only analyze Left/Right hand MI tasks
            '''
            if self.args.dataset == 'BCI_IV_2a':
                mask = (Tsession_label == 0) | (Tsession_label == 1)
                Tsession_data = Tsession_data[mask]
                Tsession_label = Tsession_label[mask]

                mask = (Esession_label == 0) | (Esession_label == 1)
                Esession_data = Esession_data[mask]
                Esession_label = Esession_label[mask]

            Tsession_data = np.concatenate((Tsession_data, Esession_data), axis=0)
            Tsession_label = np.concatenate((Tsession_label, Esession_label), axis=0)
            kfold_ACC_vals = []
            kfold_F1_vals = []
            for kfold in range(5):
                # 根据索引划分训练集和测试集
                num_val_samples = 80//5
                train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                    self.cv_split(kfold, 5, num_val_samples, Tsession_data, Tsession_label)

                train_dataloader = eegDataset(train_data, train_labels)
                val_dataloader = eegDataset(val_data, val_labels)
                test_dataloader = eegDataset(test_data, test_labels)
                model = get_model(self.args)
                if CUDA:
                    model = model.cuda()

                model_path = " "
                result_path = " "

                self.args.save_path = result_path
                acc_val, f1_val, model, pred_val, act_val = train(args=self.args, data_train=train_dataloader,
                                                                  data_val=val_dataloader, sub=sub, kfold=kfold)
                model_path = osp.join(self.args.save_path, str(self.args.model))+'/sub'+str(sub)+'_'+str(kfold)+'.pth'
                acc_val, pred_val, act_val, f1_val = test(args=self.args, test_loader=test_dataloader,
                                                          model_path=model_path)
                kfold_ACC_vals.append(acc_val)
                kfold_F1_vals.append(f1_val)
                ALL_pred_vals.extend(pred_val)
                ALL_act_vals.extend(act_val)

            ACC_vals.append(np.mean(kfold_ACC_vals))
            F1_vals.append(np.mean(kfold_F1_vals))
        ACC_vals = np.array(ACC_vals)
        F1_vals = np.array(F1_vals)
        overall_kappa = cohen_kappa_score(ALL_pred_vals, ALL_act_vals)
        txt_path = osp.join(self.args.save_path, str(self.args.model), 'result.txt')
        with open(txt_path, 'w') as f:
            # 写入 acc_vals
            f.write("ACC_vals:\n")
            for val in ACC_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("F1_vals:\n ")
            for val in F1_vals:
                f.write(f"{val}\n")
            f.write("\n")

            f.write("overall_kappa: ")
            f.write(f"{overall_kappa}\n")

        print("Finished!")

    def cv_split(self,k, k_fold, trial_num_sf, DATA, LABELS):
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