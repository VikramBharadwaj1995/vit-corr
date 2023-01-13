from tqdm import tqdm
import torch
import numpy as np

from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine

def epe(input_flow, target_flow, mean=True):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
    Output:
        Averaged end-point-error (value)
    """
    EPE = torch.norm(target_flow - input_flow, p=2, dim=1)
    if mean:
        EPE = EPE.mean()
    
    # print("Input flow: ", input_flow.shape)
    # print("Target flow: ", target_flow.shape)

    # # Numpy implementation
    # input_flow = input_flow.cpu().numpy()
    # target_flow = target_flow.cpu().numpy()
    # eucl_array = []
    # for i in range(len(input_flow)):
    #    eucl_array.append(np.linalg.norm(target_flow[i] - input_flow[i]))
    
    # print("Numpy EPE: {}".format(np.mean(eucl_array)))
    # print("Pytorch EPE: {}".format(EPE))
    return EPE


def correct_correspondences(input_flow, target_flow, alpha, img_size):
    """
    Computation PCK, i.e number of the pixels within a certain threshold
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold) # Computes dist ≤ pck_threshold element-wise (element then equal to 1)
    return mask.sum().item()


def F1_kitti_2015(input_flow, target_flow, tau=[3.0, 0.05]):
    """
    Computation number of outliers
    for which error > 3px(tau[0]) and error/magnitude(ground truth flow) > 0.05(tau[1])
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    gt_magnitude = torch.norm(target_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    mask = dist.gt(3.0) & (dist/gt_magnitude).gt(0.05)
    return mask.sum().item()


def calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range, name_dataset, rate, path_to_save=None,
                                      compute_F1=False, save=False):
    aepe_array = []
    pck_alpha_0_05_over_image = []
    pck_thresh_1_over_image = []
    pck_thresh_5_over_image = []
    F1 = 0.0

    n_registered_pxs = 0.0
    array_n_correct_correspondences = np.zeros(threshold_range.shape, dtype=np.float32)
    
    corr_list = []
    for j in range(len(test_dataloader.__dict__['dataset'].__dict__['df'])):
        string = test_dataloader.__dict__['dataset'].__dict__['df'][j]['source_image'].split("/")[-1]+"->"+test_dataloader.__dict__['dataset'].__dict__['df'][j]['target_image'].split("/")[-1]+'.npy'
        path = '/work/vig/vikram/eth3d/eth3d_model:cotr_resnet50_layer3_1024_dset:megadepth_sushi_bs:16_pe:lin_sine_lrbackbone:1e-05_suffix:stage_3_no_-1/'+name_dataset+'_every_5_rate_of_'+str(rate)+'/corr/'+string
        corr_list.append(path)

    engine = FasterSparseEngine(network, 32, mode='stretching')

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image'][0].permute(1, 2, 0)
        target_img = mini_batch['target_image'][0].permute(1, 2, 0)
        mask_gt = mini_batch['correspondence_mask'].to(device)
        flow_gt = mini_batch['flow_map'].to(device)
        Xs = mini_batch['Xs'].T
        Ys = mini_batch['Ys'].T
        Xt = mini_batch['Xt'].T
        Yt = mini_batch['Yt'].T
        if flow_gt.shape[1] != 2:
            # shape is BxHxWx2
            flow_gt = flow_gt.permute(0,3,1,2)
        bs, ch_g, h_g, w_g = flow_gt.shape
        try:
            # raw_corr = np.load(corr_list[i_batch], allow_pickle=True)
            # raw_corr = raw_corr.flat[0]['raw_corr']
            # print("Xs: ", Xs.shape)
            # print("Ys: ", Ys.shape)
            queries = np.concatenate((Xs, Ys), axis=1)
            targets = np.concatenate((Xt, Yt), axis=1)
            # print("Queries: ", queries)
            # print("Targets: ", targets)
            # print("**SHAPES BEFORE TENSOR**")
            # print("Queries: ", queries.shape)
            # print("Targets: ", targets.shape)
            cotr_out, target_idx = engine.cotr_corr_multiscale(source_img.numpy(), target_img.numpy(), queries_a=queries, force=False, return_idx=True)
            targets = targets[[list(target_idx)], :]
            targets = torch.from_numpy(targets).to(device).squeeze(0)
            cotr_out = torch.from_numpy(cotr_out[:, 2:]).to(device)
            # print("**SHAPES AFTER TENSOR**")
            # print("Cotr out: ", cotr_out.shape)
            # print("Targets: ", targets.shape)
            # print("Values")
            # print("Cotr out: ", cotr_out)
            # print("Queries: ", queries)
            # print("Targets: ", targets)
            # print("target_idx: ", target_idx)
            # print("target_idx.shape: ", target_idx.shape)
            # print("flow_estimated: ", flow_estimated)
            # print("flow_estimated shape: ", flow_estimated.shape)
            # print("queries: ", queries[[list(target_idx)], :])
            # print("targets: ", targets[[list(target_idx)], :])
            # targets = targets[[list(target_idx)], :]
            # flow_estimated = engine.cotr_corr_multiscale(source_img.numpy(), target_img.numpy(), zoom_ins=np.logspace(np.log10(0.5), np.log10(0.0625), num=4), queries_a=queries, force=False)
            # flow_estimated = cotr_corr_base(network, source_img.numpy(), target_img.numpy(), None)
            # print("flow_estimated: ", flow_estimated)
            # print("flow_target: ", flow_target)
            # flow_est = flow_estimated[:, 2:]
            # flow_est = torch.from_numpy(flow_est).to(device)
            # targets = torch.from_numpy(targets).to(device)
            # flow_target = targets.squeeze(0)
            # torch tensor of shape Bx2xH_xW_, will be the same types (cuda or cpu) depending on the device
            # H_ and W_ could be smaller than the ground truth flow (ex DCG Net takes only 240x240 images)

            # if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
            #     '''
            #     the estimated flow is downscaled (the original images were downscaled before 
            #     passing through the network)
            #     as it is the case with DCG Net, the estimate flow will have shape 240x240
            #     it needs to be upscaled to the same size as flow_target_x and rescaled accordingly:
            #     '''
            #     ratio_h = float(h_g) / float(flow_estimated.shape[2])
            #     ratio_w = float(w_g) / float(flow_estimated.shape[3])
            #     flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear',
            #                                                align_corners=False)
            #     flow_estimated[:, 0, :, :] *= ratio_w
            #     flow_estimated[:, 1, :, :] *= ratio_h
            
            # assert flow_est.shape == flow_gt.shape

            # flow_target_x = flow_gt.permute(0, 2, 3, 1)[:, :, :, 0]
            # flow_target_y = flow_gt.permute(0, 2, 3, 1)[:, :, :, 1]
            # flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 0]  # B x h_g x w_g
            # flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 1]

            # flow_target = \
            #     torch.cat((flow_target_x[mask_gt].unsqueeze(1),
            #                flow_target_y[mask_gt].unsqueeze(1)), dim=1)
            # flow_est = \
            #     torch.cat((flow_est_x[mask_gt].unsqueeze(1),
            #                flow_est_y[mask_gt].unsqueeze(1)), dim=1)
                        
            # flow_target_x[mask_gt].shape is (number of pixels), then with unsqueze(1) it becomes (number_of_pixels, 1)
            # final shape is (B*H*W , 2), B*H*W is the number of registered pixels (according to ground truth masks)

            # let's calculate EPE per batch
            # flow_target = targets
            # aepe = epe(flow_est, targets)  # you obtain the mean per pixel
            aepe = epe(cotr_out, targets)  # you obtain the mean per pixel
            aepe_array.append(aepe.item())

            # let's calculate PCK values
            # img_size = max(mini_batch['source_image_size'][0], mini_batch['source_image_size'][1]).float().to(device)
            # alpha_0_05 = correct_correspondences(flow_est, flow_target, alpha=0.05, img_size=img_size)
            # px_1 = correct_correspondences(flow_est, flow_target, alpha=1.0/float(img_size), img_size=img_size) # threshold of 1 px
            # px_5 = correct_correspondences(flow_est, flow_target, alpha=5.0/float(img_size), img_size=img_size) # threshold of 5 px

            # percentage per image is calculated for each
            # pck_alpha_0_05_over_image.append(alpha_0_05/flow_target.shape[0])
            # pck_thresh_1_over_image.append(px_1/flow_target.shape[0])
            # pck_thresh_5_over_image.append(px_5/flow_target.shape[0])
            
            print(" -- Output Values --")
            print("name of the file: ", corr_list[i_batch])
            print("Source path: ", test_dataloader.__dict__['dataset'].__dict__['df'][i_batch]['source_image'])
            print("Target path: ", test_dataloader.__dict__['dataset'].__dict__['df'][i_batch]['target_image'])
            print("AEPE:", aepe.item())
            # print("PCK_0.05:", alpha_0_05/flow_target.shape[0])
            # print("PCK_1:", px_1/flow_target.shape[0])
            # print("PCK_5:", px_5/flow_target.shape[0])
            print(" -- END --")
            # PCK curve for different thresholds ! ATTENTION, here it is over the whole dataset and not per image
            # n_registered_pxs += flow_target.shape[0]  # also equal to number of correspondences that should be correct
            # according to ground truth mask
            # for t_id, threshold in enumerate(threshold_range):
            #     array_n_correct_correspondences[t_id] += correct_correspondences(flow_est,
            #                                                                      flow_target,
            #                                                                      alpha=threshold,
            #                                                                      img_size=img_size)
                # number of correct pixel correspondence below a certain threshold, added for each batch
        except:
            pass
    output = {'final_eape': np.mean(aepe_array)}
            #   'pck_alpha_0_05_average_per_image': np.mean(pck_alpha_0_05_over_image),
            #   'pck_thresh_1_average_per_image': np.mean(pck_thresh_1_over_image),
            #   'pck_thresh_5_average_per_image': np.mean(pck_thresh_5_over_image),
            #   'alpha_threshold': threshold_range.tolist()}
            #   'pixel_threshold': np.round(threshold_range * img_size.cpu().numpy(), 2).tolist(),
            #   'pck_per_threshold_over_dataset': np.float32(array_n_correct_correspondences /
            #                                                (n_registered_pxs + 1e-6)).tolist()}

    return output