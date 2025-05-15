import torch
import torch.nn as nn
import torch.utils.data as Data

from utils.metrics import SegmentationMetricTPFNFP, ROCMetric
from utils.data import *
from model import get_segmentation_model

from tqdm import tqdm
from sklearn.metrics import auc
from argparse import ArgumentParser
from scipy.io import savemat

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Evaluation of networks')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--dataset', type=str, default='irstd', help='choose datasets')
    parser.add_argument('--sirstaug-dir', type=str, default='../sirst_aug',
                        help='dir of dataset')

    #
    # Evaluation parameters
    #
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--ngpu', type=int, default=0, help='GPU number')

    #
    # Network parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'DEA_ir_mIoU-0.6069_fmeasure-0.7554.pkl',
                        help='checkpoint path')
    parser.add_argument('--net-name', type=str, default='ERDA',
                        help='net name: ERDA')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    print('...load checkpoint: %s' % args.pkl_path)
    net = FAT_Net()
    ckpt = torch.load(args.pkl_path, map_location='cpu')
    net.load_state_dict(ckpt)
    net.to(device)
    net.eval()

    # define dataset
    if args.dataset == 'sirstaug':
        dataset = SirstAugDataset(mode='test')
    elif args.dataset == 'irstd':
            dataset = IRSTDDataset(mode='test', base_size=args.base_size)
    else:
        raise NotImplementedError
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # metrics
    metrics = SegmentationMetricTPFNFP(nclass=1)
    metric_roc = ROCMetric(nclass=1, bins=200)

    # evaluation
    tbar = tqdm(data_loader)
    for i, (data, labels) in enumerate(tbar):
        with torch.no_grad():
            data = data.to(device)
            labels = labels.to(device)
            output = net(data)

        metrics.update(labels=labels, preds=output)
        metric_roc.update(labels=labels, preds=output)

    miou, prec, recall, fmeasure = metrics.get()
    tpr, fpr = metric_roc.get()
    auc_value = auc(fpr, tpr)

    # show results
    print('dataset: %s, checkpoint: %s' % (args.dataset, args.pkl_path))
    print('Precision: %.4f | Recall: %.4f | mIoU: %.4f | F-measure: %.4f | AUC: %.4f'
          % (prec, recall, miou, fmeasure, auc_value))
    savemat(f'save/{args.dataset}_{args.net_name}_evulate.mat', {'miou': miou, 'prec': prec, 'recall': recall, 'fmeasure': fmeasure, 'auc': auc_value, 'tpr': tpr, 'fpr': fpr })

