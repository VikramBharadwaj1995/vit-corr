# Visual Correspondence using Vision Transformers
In this project we explore the use of vision transformers for the task of visual correspondence in image pairs.
We propose a novel architecture, which is an improvement on top of the original architecture proposed in COTR(ICCV 2021).

##  Training

### 1. Prepare data

See `prepare_data.md`.

### 2. Setup configuration json

Add an entry inside `COTR/global_configs/dataset_config.json`, make sure it is correct on your system. In the provided `dataset_config.json`, we have different configurations for different clusters.

Explanations on some json parameters:

`valid_list_json`: The valid list json file, see `2. Valid list` in `Scripts to generate dataset`.

`train_json/val_json/test_json`:  The splits json files, see  `3. Train/val/test split` in `Scripts to generate dataset`.

`scene_dir`: Path to Megadepth SfM folder(rectified ones!). `{0}{1}` are scene and sequence id used by f-string.

`image_dir/depth_dir`: Path to images and depth maps of Megadepth.

### 3. Example command

```python train_cotr.py --scene_file sample_data/jsons/debug_megadepth.json  --dataset_name=megadepth --info_level=rgbd --use_ram=no --batch_size=2 --lr_backbone=1e-4 --max_iter=200 --valid_iter=10 --workers=4 --confirm=no```

**Important arguments:**

`use_ram`: Set to "yes" to load data into main memory.

`crop_cam`: How to crop the image, it will change the camera intrinsic accordingly.

`scene_file`: The sequence control file.

`suffix`: Give the model a unique suffix.

`load_weights`: Load a pretrained weights, only need the model name, it will automatically find the folder with the same name under the output folder, and load the "checkpoint.pth.tar".

### 4. Our training commands

As stated in the paper, we have 3 training stages. The machine we used has 1 RTX 3090, i7-10700, and 128G RAM. We store the training data inside the main memory during the first two stages.

Stage 1: `python train_cotr.py --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./out/cotr`

Stage 2: `python train_cotr.py --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=16 --learning_rate=1e-4 --lr_backbone=1e-5 --max_iter=2000000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_2 --valid_iter=10000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./out/cotr --load_weights=model:cotr_resnet50_layer3_1024_dset:megadepth_sushi_bs:24_pe:lin_sine_lrbackbone:0.0_suffix:stage_1`

Stage 3: `python train_cotr.py --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=no --use_cc=no --batch_size=16 --learning_rate=1e-4 --lr_backbone=1e-5 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_3 --valid_iter=2000 --enable_zoom=yes --crop_cam=no_crop --out_dir=./out/cotr --load_weights=model:cotr_resnet50_layer3_1024_dset:megadepth_sushi_bs:16_pe:lin_sine_lrbackbone:1e-05_suffix:stage_2`

# Acknowledgments
I thank Dr.Huaizu Jiang for helping me throughout the project.
