import numpy as np
import os
import torch
from evaluate_new import calculate_epe_and_pck_per_dataset
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from image_transforms import ArrayToTensor
from ETH3D_interval import ETH_interval


# For COTR
import torch.nn as nn
import sys
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.models import build_model
from COTR.utils import utils


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
set_COTR_arguments(parser)
# Paths
parser.add_argument('--checkpoint_path', help='Path to the checkpoint file')
parser.add_argument('--data_dir', metavar='DIR', type=str,
                    help='path to folder containing images and flows for validation')
parser.add_argument('--save_dir', type=str, default='evaluation/',
                    help='path to directory to save the text files and results')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')
args = parser.parse_args()

torch.cuda.empty_cache()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_dict = {}
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

# define the image processing parameters, the actual pre-processing is done within the model functions
input_images_transform = transforms.Compose([ArrayToTensor(get_float=False)])
                                            # transforms.Resize((256, 256))])  # only put channel first
gt_flow_transform = transforms.Compose([ArrayToTensor()])
                                        # transforms.Resize((256, 256))])  # only put channel first
co_transform = None

# ETH3D dataset information
dataset_names = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 'delivery_area', 'electro',
                'forest', 'playground', 'terrains']

rates = list(range(3, 16, 2))

# Load COTR model
opt = parser.parse_args()
opt.command = ' '.join(sys.argv)

layer_2_channels = {'layer1': 256,
                    'layer2': 512,
                    'layer3': 1024,
                    'layer4': 2048, }
opt.dim_feedforward = layer_2_channels[opt.layer]

model = build_model(opt)
model = model.cuda()
# weights = torch.load("/home/bharadwaj.vi/COTR/out/cotr_stage21/model:cotr_resnet50_layer3_1024_dset:megadepth_bs:16_pe:lin_sine_lrbackbone:1e-05_suffix:stage_21/checkpoint.pth.tar", map_location='cpu')['model_state_dict']
weights = torch.load(args.checkpoint_path, map_location='cpu')['model_state_dict']
utils.safe_load_weights(model, weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nn.DataParallel(model)
net.eval()
network = net.to(device)

name_to_save = 'Our_COTR' + '_' + 'ETH3D'
threshold_range = np.linspace(0.002, 0.2, num=50)

dict_results = {}
for rate in rates:
    print('Computing results for interval {}...'.format(rate))
    dict_results['rate_{}'.format(rate)] = {}
    list_of_outputs_per_rate = []
    for name_dataset in dataset_names:
        print('looking at dataset {}...'.format(name_dataset))
        test_set = ETH_interval(root=args.data_dir,
                                path_list=os.path.join(args.data_dir, 'info_ETH3D_files',
                                                        '{}_every_5_rate_of_{}'.format(name_dataset, rate)),
                                source_image_transform=input_images_transform,
                                target_image_transform=input_images_transform,
                                flow_transform=gt_flow_transform,
                                co_transform=co_transform)  # only test
        test_dataloader = DataLoader(test_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=8)
        print(test_set.__len__())
        output = calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range, name_dataset, rate)
        
        # to save the intermediate results
        # dict_results['rate_{}'.format(rate)][name_dataset] = output
        list_of_outputs_per_rate.append(output)

    # average over all datasets for this particular rate of interval
    avg = {'final_eape': np.mean([list_of_outputs_per_rate[i]['final_eape'] for i in range(len(dataset_names))])}
            # 'pck_thresh_1_average_per_image': np.mean([list_of_outputs_per_rate[i]
            #                                             ['pck_thresh_1_average_per_image'] for i in range(len(dataset_names))]),
            # 'pck_thresh_5_average_per_image': np.mean([list_of_outputs_per_rate[i]
            #                                             ['pck_thresh_5_average_per_image'] for i in range(len(dataset_names))])
            # }
    dict_results['rate_{}'.format(rate)]['avg'] = avg

# save the dictionnary for this particular pre trained model
save_dict['{}'.format('Our_COTR')]=dict_results

with open('{}/{}.txt'.format(args.save_dir, 'metrics_{}'.format(name_to_save)), 'w') as outfile:
    json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
    print('written to file ')
