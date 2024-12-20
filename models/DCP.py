import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image
import math
import util.pytorch_stats_loss as stats_loss
class DCP_LOSS(nn.Module):
    def __init__(self, alpha_h, alpha_d, thresh_FOD_h, thresh_FOD_d, thresh_mask_h, thresh_mask_d):
        super(DCP_LOSS,self).__init__()

        # tradional color deconvolution for stain separation
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]).cuda()
        self.coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3,1).cuda()
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed).cuda()
        # focal FOD , alpha 
        self.alpha_d = alpha_d
        self.alpha_h = alpha_h
        self.adjust_Calibration_d = torch.tensor(10**(-(math.e)**(1/self.alpha_d))).cuda() # 
        self.adjust_Calibration_h = torch.tensor(10**(-(math.e)**(1/self.alpha_h))).cuda()
        
        # Set a threshold to identify and zero out FOD values that are too low,/
        # thereby reducing their impact on computing the tumor expression level inference.
        self.thresh_FOD_h = thresh_FOD_h
        self.thresh_FOD_d = thresh_FOD_d
        # thresh_FOD for getting pseudo mask
        self.thresh_mask_h = thresh_mask_h
        self.thresh_mask_d = thresh_mask_d

        self.log_adjust = torch.log(torch.tensor(1e-6)).cuda()
        
        self.mse_loss = nn.MSELoss().cuda()
        self.mse_loss_2 = nn.MSELoss(reduce = False).cuda()
        
    def forward(self, inputs, targets):
        # 调整输入和目标图像的形状
        inputs_reshape = inputs.permute(0, 2, 3, 1)
        targets_reshape = targets.permute(0, 2, 3, 1)

        # 分别计算第一通道和第三通道的特征
        inputs_OD_h, inputs_block_h, inputs_histo_h, inputs_mask_h, \
        inputs_OD_d, inputs_block_d, inputs_histo_d, inputs_mask_d = self.compute_OD(inputs_reshape)

        targets_OD_h, targets_block_h, targets_histo_h, targets_mask_h, \
        targets_OD_d, targets_block_d, targets_histo_d, targets_mask_d = self.compute_OD(targets_reshape)

        # 初始化损失
        loss_DCP_h = 0.0
        loss_DCP_d = 0.0

        # 计算第一通道的损失
        DCP_avg_h = self.mse_loss_2(inputs_OD_h, targets_OD_h) / (inputs.shape[2] * inputs.shape[3]) ** 2
        DCP_histo_h = (((inputs_histo_h / (inputs.shape[2] * inputs.shape[3]) - targets_histo_h / (inputs.shape[2] * inputs.shape[3])) ** 2).sum(1)) / inputs.shape[0]
        DCP_block_h = self.mse_loss((inputs_block_h / (inputs.shape[2] * inputs.shape[3] / 16)), (targets_block_h / (inputs.shape[2] * inputs.shape[3] / 16)))

        loss_DCP_h += torch.sum(torch.where(
            (inputs_OD_h - targets_OD_h >= targets_OD_h * -0.4) & (inputs_OD_h - targets_OD_h <= targets_OD_h * 0.4),
            DCP_histo_h,
            DCP_avg_h + DCP_histo_h
        ))
        loss_DCP_h += DCP_block_h

        # 计算第三通道的损失
        DCP_avg_d = self.mse_loss_2(inputs_OD_d, targets_OD_d) / (inputs.shape[2] * inputs.shape[3]) ** 2
        DCP_histo_d = (((inputs_histo_d / (inputs.shape[2] * inputs.shape[3]) - targets_histo_d / (inputs.shape[2] * inputs.shape[3])) ** 2).sum(1)) / inputs.shape[0]
        DCP_block_d = self.mse_loss((inputs_block_d / (inputs.shape[2] * inputs.shape[3] / 16)), (targets_block_d / (inputs.shape[2] * inputs.shape[3] / 16)))

        loss_DCP_d += torch.sum(torch.where(
            (inputs_OD_d - targets_OD_d >= targets_OD_d * -0.4) & (inputs_OD_d - targets_OD_d <= targets_OD_d * 0.4),
            DCP_histo_d,
            DCP_avg_d + DCP_histo_d
        ))
        loss_DCP_d += DCP_block_d

        # 合并两个通道的损失
        loss_DCP = loss_DCP_h + loss_DCP_d

        # 返回损失和掩码
        return loss_DCP, inputs_mask_h, targets_mask_h, inputs_mask_d, targets_mask_d

    
    def compute_OD(self, image):
        assert image.shape[-1] == 3  

        ihc_hed = self.separate_stains(image, self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:,:, :, 0])  

        ihc_h = self.combine_stains(torch.stack((ihc_hed[:,:, :, 0], null, null), axis=-1), self.rgb_from_hed)
        grey_h = self.rgb2gray(ihc_h)  
        grey_h[grey_h < 0.0] = torch.tensor(0.0).cuda()  
        grey_h[grey_h > 1.0] = torch.tensor(1.0).cuda()

        FOD_h = torch.log10(1 / (grey_h + self.adjust_Calibration_h))
        FOD_h[FOD_h < 0] = torch.tensor(0.0).cuda()  
        FOD_h = FOD_h ** self.alpha_h  

        FOD_h_relu = torch.where(FOD_h < self.thresh_FOD_h, torch.tensor(0.0).cuda(), FOD_h)

        mask_OD_h = torch.where(FOD_h < self.thresh_mask_h, torch.tensor(0.0).cuda(), FOD_h)
        mask_OD_h = mask_OD_h.squeeze(-1).detach()
        mask_OD_h[mask_OD_h > 0] = torch.tensor(1.0)

        flattened_img_2_h = FOD_h.squeeze(-1).flatten(1, 2)

        avg_h = torch.sum(FOD_h_relu, dim=(1, 2, 3))

        num_blocks = 16
        tensor_blocks_h = FOD_h_relu.squeeze(-1).unfold(1, image.shape[1] // int(math.sqrt(num_blocks)), image.shape[1] // int(math.sqrt(num_blocks))) \
            .unfold(2, image.shape[2] // int(math.sqrt(num_blocks)), image.shape[2] // int(math.sqrt(num_blocks)))
        block_h = tensor_blocks_h.sum(dim=(3, 4))

        num_bins = 20
        histo_h = self.calculate_histo_sums(flattened_img_2_h, num_bins, 0.0, math.e)

        ihc_d = self.combine_stains(torch.stack((null, null, ihc_hed[:,:, :, 2]), axis=-1), self.rgb_from_hed)
        grey_d = self.rgb2gray(ihc_d)  
        grey_d[grey_d < 0.0] = torch.tensor(0.0).cuda()  
        grey_d[grey_d > 1.0] = torch.tensor(1.0).cuda()

        FOD_d = torch.log10(1 / (grey_d + self.adjust_Calibration_d))
        FOD_d[FOD_d < 0] = torch.tensor(0.0).cuda()  
        FOD_d = FOD_d ** self.alpha_d  

        FOD_d_relu = torch.where(FOD_d < self.thresh_FOD_d, torch.tensor(0.0).cuda(), FOD_d)

        mask_OD_d = torch.where(FOD_d < self.thresh_mask_d, torch.tensor(0.0).cuda(), FOD_d)
        mask_OD_d = mask_OD_d.squeeze(-1).detach()
        mask_OD_d[mask_OD_d > 0] = torch.tensor(1.0)

        flattened_img_2_d = FOD_d.squeeze(-1).flatten(1, 2)

        avg_d = torch.sum(FOD_d_relu, dim=(1, 2, 3))

        tensor_blocks_d = FOD_d_relu.squeeze(-1).unfold(1, image.shape[1] // int(math.sqrt(num_blocks)), image.shape[1] // int(math.sqrt(num_blocks))) \
            .unfold(2, image.shape[2] // int(math.sqrt(num_blocks)), image.shape[2] // int(math.sqrt(num_blocks)))
        block_d = tensor_blocks_d.sum(dim=(3, 4))

        histo_d = self.calculate_histo_sums(flattened_img_2_d, num_bins, 0.0, math.e)

        return avg_h, block_h, histo_h, mask_OD_h, avg_d, block_d, histo_d, mask_OD_d

    
    def separate_stains(self,rgb, conv_matrix, *, channel_axis=-1):

        rgb = torch.maximum(rgb, torch.tensor(1e-6))  

        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.tensor(0))

        return stains
    def combine_stains(self,stains, conv_matrix, *, channel_axis=-1):

        log_rgb = -torch.matmul((stains * -self.log_adjust) , conv_matrix)
        rgb = torch.exp(log_rgb)

        return torch.clamp(rgb, min=0, max=1)
    def rgb2gray(self,rgb, *, channel_axis=-1):

        return torch.matmul(rgb ,self.coeffs)
    
    def calculate_histo_sums(self,input, num_histos, min_val, max_val):
        features = input.clone()
        bucket_width = (max_val - min_val) / num_histos

        normalized_features = (features - min_val) / bucket_width

        histo_indices = (normalized_features.clamp(0, num_histos-1)).long()

        batch_sums = torch.zeros((features.shape[0], num_histos)).cuda()

        for i in range(features.shape[0]):
            for j in range(num_histos):
                indices_in_histo = (histo_indices[i] == j).nonzero()
                if indices_in_histo.numel() > 0:
                    batch_sums[i, j] = torch.sum(features[i, indices_in_histo])
        
        return batch_sums
    
    def weighted_mse_loss(self,input,target,weights = None):
        assert input.size() == target.size()
        size = input.size()
        if weights == None:
            weights = torch.ones(size = size[0])
        
        se = ((input - target)**2)
        
        return (se*weights).mean()
    

