from two_stage import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='BlueBCI',choices=["BCI_IV_2a","BlueBCI"])
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--sampling-rate', type=int, default=250)
    parser.add_argument('--input-shape', type=tuple, default=(1, 8, 1000))
    parser.add_argument('--data-format', type=str, default='eeg')

    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--patient', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=9,choices='9')
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--save-path', default='F://Result/')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=str, default='IGNNNet_stu',
                        choices=['IGNNNet','IGNNNet_stu'])

    args = parser.parse_args()
    sub_to_run = np.arange(1, 2)
    TW = two_stage(args)
    TW.run(sub_to_run, data_type="data")
    # TW.run_kfold(sub_to_run, data_type="data")
    # TW.KU_transfer_TargetDataset_no_finetuning(sub_to_run, data_type="data")
    # TW.KU_transfer_TargetDataset_finetuning(sub_to_run, data_type="data")
    # TW.KU_transfer_TargetDataset_stu_train(sub_to_run, data_type="data")
    # TW.KU_transfer_TargetDataset_stu_train_kfold(sub_to_run, data_type="data")
    print('end')