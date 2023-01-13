import numpy as np
import os
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import argparse
import time
import fnmatch
from data.dataset import HPatchesDataset
from evaluate import calculate_epe_hpatches, calculate_pck_hpatches
from COTR.options.options import *
from COTR.options.options_utils import *

from COTR.models import build_model
from COTR.utils import utils

# Argument parsing
parser = argparse.ArgumentParser()
set_COTR_arguments(parser)
# Paths
parser.add_argument('--csv-path', type=str, default='data/csv',
                    help='path to training transformation csv folder')
parser.add_argument('--image-data-path', type=str,
                    default='data/hpatches-geometry',
                    help='path to folder containing training images')
# parser.add_argument('--model', type=str, default='dgc',
#                     help='Model to use', choices=['dgc', 'dgcm'])
# parser.add_argument('--metric', type=str, default='aepe',
#                     help='Model to use', choices=['aepe', 'pck'])
parser.add_argument('--batch-size', type=int, default=1,
                    help='evaluation batch size')
# parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

opt = parser.parse_args()
opt.command = ' '.join(sys.argv)

layer_2_channels = {'layer1': 256,
                    'layer2': 512,
                    'layer3': 1024,
                    'layer4': 2048, }
opt.dim_feedforward = layer_2_channels[opt.layer]

# Image normalisation
mean_vector = np.array([0.485, 0.456, 0.406])
std_vector = np.array([0.229, 0.224, 0.225])
normTransform = transforms.Normalize(mean_vector, std_vector)
dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

model = build_model(opt)
model = model.cuda()
weights = torch.load("/home/bharadwaj.vi/COTR/out/cotr/model:cotr_resnet50_layer3_1024_dset:megadepth_bs:24_pe:lin_sine_lrbackbone:0.0_suffix:stage_1/checkpoint.pth.tar", map_location='cpu')['model_state_dict']
# weights = torch.load("/home/bharadwaj.vi/visual-correspondence/COTR/out/default/checkpoint.pth.tar", map_location='cpu')['model_state_dict']
utils.safe_load_weights(model, weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nn.DataParallel(model)
net.eval()
net = net.to(device)

with torch.no_grad():
    number_of_scenes = 5
    res = []
    jac = []
    # create a threshold range
    threshold_range = np.linspace(1/256, 5/256, num=3)
    res_pck = np.zeros((number_of_scenes, len(threshold_range)))

    # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
    for id, k in enumerate(range(2, number_of_scenes + 2)):
        test_dataset = \
            HPatchesDataset(csv_file=osp.join(opt.csv_path,
                                              'hpatches_1_{}.csv'.format(k)),
                            image_path_orig=opt.image_data_path,
                            transforms=dataset_transforms)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        epe_arr = calculate_epe_hpatches(net,
                                            test_dataloader,
                                            device)
        res.append(np.mean(epe_arr))
        print("(AEPE for batch) -> Scene {}: {}".format(k, res[-1]))
        for t_id, threshold in enumerate(threshold_range):
            res_pck[id, t_id] = calculate_pck_hpatches(net,
                                                    test_dataloader,
                                                    device,
                                                    alpha=threshold)
            print("(PCK for 1) -> Scene {}: {}".format(k, res_pck[id, t_id]))

    print("AEPE results: {}".format(res))
    print("PCK results: {}".format(res_pck))
