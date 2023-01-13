import os
import math
import os.path as osp
import time

import tqdm
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image, ImageDraw


from COTR.utils import utils, debug_utils, constants
from COTR.trainers import base_trainer, tensorboard_helper
from COTR.projector import pcd_projector


class COTRTrainer(base_trainer.BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion,
                 train_loader, val_loader):
        super().__init__(opt, model, optimizer, criterion,
                         train_loader, val_loader)

    def validate_batch(self, data_pack):
        assert self.model.training is False
        with torch.no_grad():
            img = data_pack['image'].cuda()
            query = data_pack['queries'].cuda()
            target = data_pack['targets'].cuda()
            self.optim.zero_grad()
            pred = self.model(img, query, 'normal')['pred_corrs']
            
            # loss = torch.nn.functional.mse_loss(pred, target)
            ## Adding auxillary loss
            
            loss = 0
            for aux_pred in pred:
                loss += torch.nn.functional.mse_loss(aux_pred, target)
            
            aux_pred = pred[-1]
            if self.opt.cycle_consis and self.opt.bidirectional:
                cycle_error = self.model(img, aux_pred, 'cyclic')['pred_corrs']
                # mask = torch.norm(cycle_error[-1] - query, dim=-1) < 10 / constants.MAX_SIZE
                cycle_loss = 0
                for cycle_aux_pred in cycle_error:
                    cycle_loss += torch.nn.functional.mse_loss(cycle_aux_pred, query)
                loss += cycle_loss
            elif self.opt.cycle_consis and not self.opt.bidirectional:
                img_reverse = torch.cat([img[..., constants.MAX_SIZE:], img[..., :constants.MAX_SIZE]], axis=-1)
                query_reverse = pred.clone()
                query_reverse[..., 0] = query_reverse[..., 0] - 0.5
                cycle = self.model(img_reverse, query_reverse)['pred_corrs']
                cycle[..., 0] = cycle[..., 0] - 0.5
                mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                if mask.sum() > 0:
                    cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                    loss += cycle_loss
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                print('loss is nan while validating')
            
            # Calculating validation accuracy
            aux_pred = pred[-1]
            aux_pred_temp = aux_pred.clone().cpu()
            target_temp = target.clone().cpu()
            correct = []
            for ap, tar in zip(aux_pred_temp, target_temp):
                correct_val = 0
                tar *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE])
                ap *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE])
                for a, t in zip(ap, tar):
                    if np.linalg.norm(np.array(a.cpu()) - np.array(t.cpu())) <= 3:
                        correct_val += 1
                # print("Correct val - ", correct_val)
                correct.append(correct_val/len(ap))
            accuracy = np.mean(correct)
            # print(" ** Accuracy : **", accuracy)

            # Calculating validation AEPE
            val_aepe = 0
            for ap, tar in zip(aux_pred, target):
                val_aepe += torch.norm(ap - tar, p=2, dim=-1).mean().item()
            val_aepe /= len(aux_pred)
            # print(" ** AEPE : **", val_aepe, len(aux_pred))
            return loss_data, aux_pred, accuracy, val_aepe 

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.model.training
        self.model.eval()
        val_loss_list = []
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            loss_data, pred, accuracy, val_aepe = self.validate_batch(data_pack)
            val_loss_list.append(loss_data)
        mean_loss = np.array(val_loss_list).mean()
        validation_data = {'val_loss': mean_loss,
                           'pred': pred,
                           'accuracy': accuracy,
                            'val_aepe': val_aepe,
                           }
        self.push_validation_data(data_pack, validation_data)
        self.save_model()
        if training:
            self.model.train()

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if self.iteration % (10 * self.valid_iter) == 0:
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
            }, osp.join(self.out, f'{self.iteration}_checkpoint.pth.tar'))

    def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        for img, corr in zip(imgs, corrs):
            img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            # corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            for c in corr:
                draw.line(c, fill=col)
            out.append(np.array(img))
        out = np.array(out) / 255.0
        return utils.np_img_to_torch_img(out)

    def draw_corrs_correct(self, imgs, pred_corrs, gt_corrs):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        
        for img, corr, gt_corr in zip(imgs, pred_corrs, gt_corrs):
            img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            gt_corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            for c, gtc in zip(corr, gt_corr):
                if np.linalg.norm(np.array(c[2:]) - np.array(gtc[2:])) <= 3:
                    draw.line(c, fill=(0, 255, 0))
                else:
                    draw.line(c, fill=(255, 0, 0))
            out.append(np.array(img))
        out = np.array(out) / 255.0
        return utils.np_img_to_torch_img(out)

    def push_validation_data(self, data_pack, validation_data):
        val_loss = validation_data['val_loss']
        val_accuracy = validation_data['accuracy']
        pred_corrs = np.concatenate([data_pack['queries'].numpy(), validation_data['pred'].cpu().numpy()], axis=-1)
        gt_corrs = np.concatenate([data_pack['queries'].numpy(), data_pack['targets'].cpu().numpy()], axis=-1)
        pred_img = self.draw_corrs_correct(data_pack['image'], pred_corrs, gt_corrs)
        gt_img = self.draw_corrs(data_pack['image'], gt_corrs, (0, 255, 0))

        gt_img = vutils.make_grid(gt_img, normalize=True, scale_each=True)
        pred_img = vutils.make_grid(pred_img, normalize=True, scale_each=True)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_scalar({'loss/val': val_loss})
        tb_datapack.add_image({'image/gt_corrs': gt_img})
        tb_datapack.add_image({'image/pred_corrs': pred_img})
        tb_datapack.add_scalar({'accuracy/val': val_accuracy})
        tb_datapack.add_scalar({'aepe/val': validation_data['val_aepe']})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
        with torch.autograd.set_detect_anomaly(True):
            img = data_pack['image'].cuda()
            query = data_pack['queries'].cuda()
            target = data_pack['targets'].cuda()
            self.optim.zero_grad()
            pred = self.model(img, query, 'normal')['pred_corrs']
            
            ## Adding auxillary loss
            loss = 0
            for aux_pred in pred:
                loss += torch.nn.functional.mse_loss(aux_pred, target)
            if self.opt.cycle_consis and self.opt.bidirectional:
                cycle_error = self.model(img, aux_pred, 'cycle')['pred_corrs']
                # mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                # if mask.sum() > 0:
                cycle_loss = 0
                for cycle_aux_pred in cycle_error:
                    cycle_loss += torch.nn.functional.mse_loss(cycle_aux_pred, query)
                loss += cycle_loss
            elif self.opt.cycle_consis and not self.opt.bidirectional:
                    img_reverse = torch.cat([img[..., constants.MAX_SIZE:], img[..., :constants.MAX_SIZE]], axis=-1)
                    query_reverse = pred.clone()
                    query_reverse[..., 0] = query_reverse[..., 0] - 0.5
                    cycle = self.model(img_reverse, query_reverse)['pred_corrs']
                    cycle[..., 0] = cycle[..., 0] - 0.5
                    mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                    if mask.sum() > 0:
                        cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                        loss += cycle_loss
            loss_data = loss.data.item()

            # Calculating train accuracy
            aux_pred = pred[-1]
            correct = []
            aux_pred_temp = aux_pred.clone().detach().cpu()
            target_temp = target.clone().detach().cpu()
            for ap, tar in zip(aux_pred_temp, target_temp):
                correct_val = 0
                tar *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE])
                ap *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE])
                for a, t in zip(ap, tar):
                    if np.linalg.norm(np.array(a.detach().cpu()) - np.array(t.detach().cpu())) <= 3:
                        correct_val += 1
                # print("Correct val: ", correct_val)
                correct.append(correct_val // len(ap))
            accuracy = np.mean(correct)
            # print(" ** Accuracy : **", accuracy)
            
            # Calculating training AEPE
            train_aepe = 0
            for ap, tar in zip(aux_pred, target):
                train_aepe += torch.norm(ap - tar, p=2, dim=-1).mean().item()
            train_aepe /= len(aux_pred)
            # print(" ** AEPE : **", train_aepe, len(aux_pred))

            if np.isnan(loss_data):
                print('loss is nan during training')
                self.optim.zero_grad()
            else:
                loss.backward()
                self.push_training_data(data_pack, pred, target, loss, accuracy, train_aepe)
            self.optim.step()

    def push_training_data(self, data_pack, pred, target, loss, accuracy, train_aepe):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_histogram({'distribution/pred': pred})
        tb_datapack.add_histogram({'distribution/target': target})
        tb_datapack.add_scalar({'loss/train': loss})
        tb_datapack.add_scalar({'accuracy/train': accuracy})
        tb_datapack.add_scalar({'aepe/train': train_aepe})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.

        Arguments:
            opt {[type]} -- [description]
        '''
        if hasattr(self.opt, 'load_weights'):
            assert self.opt.load_weights is None or self.opt.load_weights == False
        # 1. load check point
        checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(
                'model check point cannnot found: {0}'.format(checkpoint_path))
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def load_pretrained_weights(self):
        '''
        load pretrained weights from another model
        '''
        # if hasattr(self.opt, 'resume'):
        #     assert self.opt.resume is False
        assert os.path.isfile(self.opt.load_weights_path), self.opt.load_weights_path

        saved_weights = torch.load(self.opt.load_weights_path)['model_state_dict']
        utils.safe_load_weights(self.model, saved_weights)
        content_list = []
        content_list += [f'Loaded pretrained weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)