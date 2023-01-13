from joblib import PrintTime
from tqdm import tqdm
import itertools
import torch
import numpy as np
from COTR.utils.constants import MAX_SIZE
from COTR.models.position_encoding import build_position_encoding

def epe(input_flow, target_flow):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxWx2]
        target_flow: ground-truth flow [BxHxWx2]
    Output:
        Averaged end-point-error (value)
    """
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


def correct_correspondences(input_flow, target_flow, alpha, img_size=256):
    """
    Computation PCK, i.e number of the pixels within a certain threshold
    Args:
        input_flow: estimated flow [BxHxWx2]
        target_flow: ground-truth flow [BxHxWx2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    input_flow = input_flow.unsqueeze(0)
    target_flow = target_flow.unsqueeze(0)
    dist = torch.norm(target_flow - input_flow, p=2, dim=0).unsqueeze(0)
    pck_threshold = alpha * img_size
    pck = torch.sum(dist < pck_threshold).float() / dist.shape[0]
    return pck


def calculate_epe_hpatches(net, val_loader, device, img_size=256):
    """
    Compute EPE for HPatches dataset
    Args:
        net: trained model
        val_loader: input dataloader
        device: `cpu` or `gpu`
        img_size: size of input images
    Output:
        aepe_array: averaged EPE for the whole sequence of HPatches
    """
    aepe_array = []
    n_registered_pxs = 0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for _, mini_batch in pbar:
        source_img = mini_batch['source_image'].to(device)
        target_img = mini_batch['target_image'].to(device)
        bs, _, _, _ = source_img.shape

        # Form the queries
        cat_stack = torch.cat((source_img, target_img), dim=-1)
        q_list = []
        for i in range(MAX_SIZE):
            queries = []
            for j in range(MAX_SIZE):
                queries.append([(i) / (MAX_SIZE), j / MAX_SIZE])
            queries = np.array(queries)
            q_list.append(queries)

        queries = torch.from_numpy(np.concatenate(q_list))[None].float().to(device)
        # net prediction
        estimates_grid = net(cat_stack, queries, 'normal')
        flow_est = estimates_grid['pred_corrs'][-1].to(device)
        flow_target = torch.sigmoid(mini_batch['correspondence_map']).to(device)
        flow_est = flow_est.reshape(bs, img_size, img_size, 2)

        # applying mask
        mask_x_gt = \
            flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = \
            flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3),
                             mask_xx_gt.unsqueeze(3)), dim=3)
        
        for i in range(bs):
            # unnormalize the flow: [-1; 1] -> [0; im_size - 1]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / (1 + 1)
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1) / (1 + 1)
        
        flow_target_x = flow_target[:, :, :, 0]
        flow_target_y = flow_target[:, :, :, 1]
        flow_est_x = flow_est[:, :, :, 0]
        flow_est_y = flow_est[:, :, :, 1]

        flow_target = \
            torch.cat((flow_target_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                       flow_target_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)
        flow_est = \
            torch.cat((flow_est_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                       flow_est_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)

        # print("flow_est max : ", flow_est.max())
        # print("flow_target max: ", flow_target.max())
        # print("flow_est min : ", flow_est.min())
        # print("flow_target min: ", flow_target.min())
        # print("-----------------------------------------------------")
        # let's calculate EPE
        aepe = epe(flow_est, flow_target)
        aepe_array.append(aepe.item())
        n_registered_pxs += flow_target.shape[0]

    return aepe_array


def calculate_pck_hpatches(net, val_loader, device, alpha=1, img_size=256):
    """
    Compute PCK for HPatches dataset
    Args:
        net: trained model
        val_loader: input dataloader
        device: `cpu` or `gpu`
        alpha: threshold to compute PCK
        img_size: size of input images
    Output:
        pck: pck value for the whole sequence of HPatches for a
            particular threhold `alpha`
    """
    n_correspondences = 0
    n_correct_correspondences = 0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for _, mini_batch in pbar:

        source_img = mini_batch['source_image'].to(device)
        target_img = mini_batch['target_image'].to(device)

        # Form the queries
        cat_stack = torch.cat((source_img, target_img), dim=-1)
        q_list = []
        for i in range(MAX_SIZE):
            queries = []
            for j in range(MAX_SIZE):
                queries.append([(j) / (MAX_SIZE), i / MAX_SIZE])
            queries = np.array(queries)
            q_list.append(queries)

        queries = torch.from_numpy(np.concatenate(q_list))[None].float().to(device)
        # network estimates
        estimates_grid = net(cat_stack, queries, 'normal')

        flow_est = estimates_grid['pred_corrs'][-1].to(device)
        flow_target = torch.sigmoid(mini_batch['correspondence_map']).to(device)
        bs, ch_g, h_g, w_g = flow_target.shape
        flow_est = flow_est.reshape(bs, img_size, img_size, 2)

        # applying mask
        mask_x_gt = \
            flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = \
            flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3),
                             mask_xx_gt.unsqueeze(3)), dim=3)

        for i in range(bs):
            # unnormalize the flow: [-1; 1] -> [0; im_size - 1]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / (1 + 1)
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1) / (1 + 1)

        flow_target = flow_target.contiguous().view(1, bs * h_g * w_g * ch_g)
        flow_est = flow_est.contiguous().view(1, bs * h_g * w_g * ch_g)
        # applying mask
        flow_target_m = \
            flow_target[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]
        flow_est_m = \
            flow_est[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]

        n_correspondences += len(flow_target_m)
        n_correct_correspondences += correct_correspondences(flow_est_m,
                                                             flow_target_m,
                                                             alpha=alpha,
                                                             img_size=img_size)
    print("n_correspondences: ", n_correspondences)
    print("n_correct_correspondences: ", n_correct_correspondences)
    pck = n_correct_correspondences / (n_correspondences + 1e-6)
    return pck