import numpy as np
import os
import torch
from evaluate_new import calculate_epe_and_pck_per_dataset
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from image_transforms import ArrayToTensor
from tqdm import tqdm
import torch.nn as nn
import sys
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.models import build_model
from COTR.utils import utils
import dataset

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

model = build_model(opt)
model = model.cuda()
weights = torch.load("/home/bharadwaj.vi/COTR/out/cotr/model:cotr_resnet50_layer3_1024_dset:megadepth_bs:24_pe:lin_sine_lrbackbone:0.0_suffix:stage_1/checkpoint.pth.tar", map_location='cpu')['model_state_dict']
utils.safe_load_weights(model, weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nn.DataParallel(model)
net.eval()
net = net.to(device)

# define the image processing parameters, the actual pre-processing is done within the model functions
input_images_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first
gt_flow_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
co_transform = None

with torch.no_grad():
    # Datasets with ground-truth flow fields available

    # HPATCHES dataset
    threshold_range = np.linspace(0.002, 0.2, num=50)
    number_of_scenes = 5
    list_of_outputs = []

    # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
    for id, k in enumerate(range(2, number_of_scenes + 2)):
        # looks at each scene individually
        test_set = dataset.HPatchesdataset(root=opt.image_data_path, csv=os.path.join(opt.csv_path,
                                              'hpatches_1_{}.csv'.format(k)),
                                            image_transform=input_images_transform, flow_transform=gt_flow_transform, 
                                            co_transform=co_transform)

        test_dataloader = DataLoader(test_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=8)

        output_scene = calculate_epe_and_pck_per_dataset(test_dataloader, net, device, threshold_range)
        list_of_outputs.append(output_scene)

    output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2],
                'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[4]}
    
    print("Outputs: ", output)
    
    with open('/home/bharadwaj.vi/COTR/results_our_model.json', 'w') as f:
        json.dump(output, f)
    print("Results saved to /home/bharadwaj.vi/COTR/out/default/results.json")
