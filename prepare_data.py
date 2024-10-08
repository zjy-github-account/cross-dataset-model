# This is the processing script of DEAP dataset
import mne
import pandas as pd
import scipy.signal
import scipy.io as sciio
from train_model import *
from sklearn import preprocessing
from scipy import signal


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.selected_data = [1]
        # self.selected_data = [1, 2, 3, 5, 6, 7, 8, 9, 15, 18, 21, 22, 28, 29, 33, 35, 36, 37, 39, 43, 44, 45]
        self.channel_8 = [10,14,4,12,7,20,13,17] # "FC6,C4,Fz,C3,FC5,CP6,Cz,CP5"
    def run(self, sub_to_run, split=False):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>/sub0.hdf'
        """
        for sub in sub_to_run:
            data_, label_ = self.load_data_one_subject(sub)
            label_ = label_.squeeze() - 1

            data_ = self.preprocess(data_)
            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def load_data_one_subject(self, sub):
        def down_sample(data, original_sample, aim_sample, type):
            step = int(original_sample / aim_sample)
            epochs = int(data.shape[2] / step)
            trials = data.shape[0]
            Channels = data.shape[1]
            New_data = np.zeros((trials, Channels, epochs))
            for i in range(trials):
                for j in range(epochs):
                    if type == 1:
                        New_data[i, :, j] = np.mean(data[i, :, j * step:(j + 1) * step], 1)
                    if type == 2:
                        New_data[i, :, j] = np.max(data[i, :, j * step:(j + 1) * step], 1)
                    if type == 3:
                        New_data[i, :, j] = data[i, :, j * step]
            return New_data

        raw_train_session1 = sciio.loadmat(self.data_path + '/Traindata' + str(sub) + '.mat')['smt']
        raw_test_session1 = sciio.loadmat(self.data_path + '/Testdata' + str(sub) + '.mat')['smt']
        raw_train_class_session1 = sciio.loadmat(self.data_path + '/Trainclass' + str(sub) + '.mat')['class']
        raw_test_class_session1 = sciio.loadmat(self.data_path + '/Testclass' + str(sub) + '.mat')['class']
        raw_train_session2 = sciio.loadmat(self.data_path + '2/Traindata' + str(sub) + '.mat')['smt']
        raw_test_session2 = sciio.loadmat(self.data_path + '2/Testdata' + str(sub) + '.mat')['smt']
        raw_train_class_session2 = sciio.loadmat(self.data_path + '2/Trainclass' + str(sub) + '.mat')['class']
        raw_test_class_session2 = sciio.loadmat(self.data_path + '2/Testclass' + str(sub) + '.mat')['class']
        train_data_session1 = raw_train_session1.transpose(1, 2, 0)
        test_data_session1 = raw_test_session1.transpose(1, 2, 0)
        train_data_session2 = raw_train_session2.transpose(1, 2, 0)
        test_data_session2 = raw_test_session2.transpose(1, 2, 0)
        train_data = np.concatenate((train_data_session1, test_data_session1), axis=0)
        test_data = np.concatenate((train_data_session2, test_data_session2), axis=0)
        train_class = np.concatenate((raw_train_class_session1, raw_test_class_session1), axis=1)
        test_class = np.concatenate((raw_train_class_session2, raw_test_class_session2), axis=1)
        DATA = np.concatenate((train_data, test_data), axis=0)
        CLASS = np.concatenate((train_class.T, test_class.T), axis=0)
        DATA = down_sample(DATA, 1000, 250, type=3)
        # chans_motor = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]  # 20 中央运动脑区
        # DATA = DATA[:, np.array(chans_motor), :].squeeze()
        # reorder the EEG channel to build the local-global graphs
        # DATA = self.reorder_channel(data=DATA, graph=self.graph_type)
        # print('data:' + str(DATA.shape) + ' label:' + str(CLASS.shape))
        return DATA, CLASS

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def preprocess(self, data, band=None, N_Flag=False):
        def filter(DATA, band=None, frequency=250):
            n_trials, n_channels = DATA.shape[0], DATA.shape[1]
            b, a = scipy.signal.butter(3, [band[0] / (frequency / 2), band[1] / (frequency / 2)], btype='bandpass')
            # 滤波和标准化(Z-SCORE)
            result = DATA.copy()
            result = result.squeeze()
            for ss in range(n_trials):
                for yy in range(n_channels):
                    data_temp1 = result[ss, yy, :]
                    data_temp1 = scipy.signal.filtfilt(b, a, data_temp1)
                    result[ss, yy, :] = data_temp1
            return result
        def z_score(data):  # Z-Score标准化/
            DATA = data.copy()
            DATA = np.squeeze(DATA)
            x = np.size(DATA, 0)
            for k in range(x):
                DATA[k] = preprocessing.scale(DATA[k])
            # DATA = np.expand_dims(DATA, axis=3)
            return DATA
        def detrend(data):  #
            DATA = data.copy()
            x = np.size(DATA, 0)
            y = np.size(DATA, 1)
            for k in range(x):
                for i in range(y):
                    datatemp = DATA[k, i, :]
                    DATA[k, i, :] = scipy.signal.detrend(datatemp)
            return DATA

        data = detrend(data)
        if band != None:
            data = filter(data, band)
        if N_Flag == True:
            data = z_score(data)
        data = np.expand_dims(data, axis=1)
        return data

    def load_data_one_subject_BlueBCI(self, sub_to_run):
        path = self.args.data_path + '蓝色传感22人数据new//'
        for sub in sub_to_run:
            DATA, LABELS = [], []
            for i in range(5):
                data = pd.read_csv(path + str(sub) + '//offLineData//block' + str(i + 1) + '.csv')
                data = np.array(data)
                for index in range(len(data)):
                    if int(data[index, 10]) in [1, 2]:
                        DATA.append(data[int(index + 250 * 1.5):int(index + 250 * 5.5), 0:8])
                        LABELS.append(data[index, 10])
            DATA = np.stack(DATA, axis=0)
            DATA = DATA.transpose(0, 2, 1)
            LABELS = np.stack(LABELS, axis=0)
            LABELS = LABELS.squeeze() - 1
            data_raw = self.preprocess(DATA, N_Flag=True) * 1e-3
            data_4_40 = self.preprocess(DATA, [4, 40], N_Flag=True) * 1e-3
            data_0_40 = self.preprocess(DATA, [0.01, 40], N_Flag=True) * 1e-3
            data_0_38 = self.preprocess(DATA, [0.01, 38], N_Flag=True) * 1e-3
            data_0_100 = self.preprocess(DATA, [0.01, 100], N_Flag=True) * 1e-3

            session_split_index = int(len(data_raw) * 0.6)

            save_path = os.getcwd()
            data_type = 'data_{}_{}'.format(self.args.data_format, str(self.args.dataset))
            save_path = osp.join(save_path, data_type,'T')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                pass
            name = 'sub' + str(sub) + '.hdf'
            save_path = osp.join(save_path, name)
            dataset = h5py.File(save_path, 'w')

            dataset['data'] = data_raw[:session_split_index]
            dataset['data_4_40'] = data_4_40[:session_split_index]
            dataset['data_0_40'] = data_0_40[:session_split_index]
            dataset['data_0_38'] = data_0_38[:session_split_index]
            dataset['data_0_100'] = data_0_100[:session_split_index]
            dataset['label'] = LABELS[:session_split_index]
            dataset.close()

            save_path = os.getcwd()
            data_type = 'data_{}_{}'.format(self.args.data_format, str(self.args.dataset))
            save_path = osp.join(save_path, data_type, 'E')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                pass
            name = 'sub' + str(sub) + '.hdf'
            save_path = osp.join(save_path, name)
            dataset = h5py.File(save_path, 'w')

            dataset['data'] = data_raw[session_split_index:]
            dataset['data_4_40'] = data_4_40[session_split_index:]
            dataset['data_0_40'] = data_0_40[session_split_index:]
            dataset['data_0_38'] = data_0_38[session_split_index:]
            dataset['data_0_100'] = data_0_100[session_split_index:]
            dataset['label'] = LABELS[session_split_index:]
            dataset.close()

    def load_data_one_subject_BCIIV2A(self, sub_to_run):
        for sub in sub_to_run:
            DATA, LABELS = self.get_epochs_T(self.args.data_path,sub)
            DATA = DATA.get_data()
            LABELS = LABELS.squeeze() - 1
            data_raw = self.preprocess(DATA, N_Flag=True)
            data_4_40 = self.preprocess(DATA, [4, 40], N_Flag=True)
            data_0_40 = self.preprocess(DATA, [0.01, 40], N_Flag=True)
            data_0_38 = self.preprocess(DATA, [0.01, 38], N_Flag=True)
            data_0_100 = self.preprocess(DATA, [0.01, 100], N_Flag=True)

            save_path = os.getcwd()
            data_type = 'data_{}_{}'.format(self.args.data_format, str(self.args.dataset))
            save_path = osp.join(save_path, data_type,'T')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                pass
            name = 'sub' + str(sub) + '.hdf'
            save_path = osp.join(save_path, name)
            dataset = h5py.File(save_path, 'w')

            dataset['data'] = data_raw
            dataset['data_4_40'] = data_4_40
            dataset['data_0_40'] = data_0_40
            dataset['data_0_38'] = data_0_38
            dataset['data_0_100'] = data_0_100
            dataset['label'] = LABELS
            dataset.close()

            DATA, LABELS = [], []

            DATA, LABELS = self.get_epochs_E(self.args.data_path, sub)
            DATA = DATA.get_data()
            LABELS = LABELS.squeeze() - 1
            data_raw = self.preprocess(DATA, N_Flag=True)
            data_4_40 = self.preprocess(DATA, [4, 40], N_Flag=True)
            data_0_40 = self.preprocess(DATA, [0.01, 40], N_Flag=True)
            data_0_38 = self.preprocess(DATA, [0.01, 38], N_Flag=True)
            data_0_100 = self.preprocess(DATA, [0.01, 100], N_Flag=True)

            save_path = os.getcwd()
            data_type = 'data_{}_{}'.format(self.args.data_format, str(self.args.dataset))
            save_path = osp.join(save_path, data_type, 'E')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                pass
            name = 'sub' + str(sub) + '.hdf'
            save_path = osp.join(save_path, name)
            dataset = h5py.File(save_path, 'w')

            dataset['data'] = data_raw
            dataset['data_4_40'] = data_4_40
            dataset['data_0_40'] = data_0_40
            dataset['data_0_38'] = data_0_38
            dataset['data_0_100'] = data_0_100
            dataset['label'] = LABELS
            dataset.close()

    def get_epochs_T(self, path, subject_num):
        raw = mne.io.read_raw_gdf(path + '//A0' + str(subject_num) + 'T.gdf')
        # Find the events time positions
        events, events_dict = mne.events_from_annotations(raw)
        # Remove the EOG channels and pick only desired EEG channels
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                               exclude='bads')
        # Extracts epochs of 3s time period from the datset into 288 events for all 4 classes
        tmin, tmax = 2., 5.996
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        if subject_num == 4:
            event_id = dict({'768': 4})
        else:
            event_id = dict({'768': 6})
        # event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
        # event_id = dict({'769': 5, '770': 6, '771': 7, '772': 8})
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
        # Getting labels and changing labels from 7,8,9,10 -> 1,2,3,4
        # labels = epochs.events[:, -1] - 6
        labels = sciio.loadmat(path + '/true labels/A0' + str(subject_num) + 'T.mat')['classlabel']
        return epochs, np.array(labels)

    def get_epochs_E(self, path, subject_num):
        raw = mne.io.read_raw_gdf(path + '/A0' + str(subject_num) + 'E.gdf')
        # Find the events time positions
        events, events_dict = mne.events_from_annotations(raw)
        # Remove the EOG channels and pick only desired EEG channels
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                               exclude='bads')
        # Extracts epochs of 3s time period from the datset into 288 events for all 4 classes
        tmin, tmax = 2., 5.996
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        event_id = dict({'768': 6})
        # event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
        # event_id = dict({'769': 5, '770': 6, '771': 7, '772': 8})
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
        # Getting labels and changing labels from 7,8,9,10 -> 1,2,3,4
        # labels = epochs.events[:, -1] - 6
        labels = sciio.loadmat(path + '/true labels/A0' + str(subject_num) + 'E.mat')['classlabel']
        return epochs, np.array(labels)

    '''
    22-channel names：'Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz',
                        'C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'.
    Related chan_num in KU dataset: [4,32,8,9,33,34,12,35,13,
                                    36,14,37,38,18,39,19,40,41,24,42,43]    #21channel, no 'FCz'
    '''