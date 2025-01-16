import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
# Nucleus-Membrane Cross-Channel Correlation
class NMCC_Loss(nn.Module):
    def __init__(self, extractor,alpha_h,alpha_d):
        super(ChannelCorrelationLoss, self).__init__()
        self.extractor = extractor  
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]).cuda()
        self.coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3,1).cuda()
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed).cuda()
        # focal FOD , alpha 
        self.alpha_h = alpha_h 
        self.alpha_d = alpha_d
        self.adjust_Calibration_h = torch.tensor(10**(-(math.e)**(1/self.alpha_h))).cuda() # 
        self.adjust_Calibration_d = torch.tensor(10**(-(math.e)**(1/self.alpha_d))).cuda()
        


        self.log_adjust = torch.log(torch.tensor(1e-6)).cuda()

    def forward(self, input, target):

        input_h, input_d = self.get_rgb(input)
        target_h, target_d = self.get_rgb(target)

        input_h = self.transform(input_h)
        input_d = self.transform(input_d)
        target_h = self.transform(target_h)
        target_d = self.transform(target_d)

        with torch.no_grad():
            input_h_f = self.extractor(input_h).squeeze()  
            input_d_f = self.extractor(input_d).squeeze()  
            target_h_f = self.extractor(target_h).squeeze()  
            target_d_f = self.extractor(target_d).squeeze()  


        input_cos = self.compute_cos(input_h_f, input_d_f)
        target_cos = self.compute_cos(target_h_f, target_d_f)
        return abs(input_cos - target_cos).mean()


    def get_rgb(self, image):

        image = image.permute(0, 2, 3, 1)

        assert image.shape[-1] == 3 

        ihc_hed = self.separate_stains(image, self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:,:, :, 0])  

        ihc_chanel_h = self.combine_stains(torch.stack((ihc_hed[:,:, :, 0], null, null), axis=-1), self.rgb_from_hed)
        ihc_chanel_d = self.combine_stains(torch.stack((null, null, ihc_hed[:,:, :, 2]), axis=-1), self.rgb_from_hed)
        
        h = ihc_chanel_h.permute(0, 3, 1, 2)
        d = ihc_chanel_d.permute(0, 3, 1, 2)


        return h, d


    def compute_cos(self, v_a, v_b):

        cosine_similarity = F.cosine_similarity(v_a, v_b, dim=-1)  

        return cosine_similarity

    def separate_stains(self,rgb, conv_matrix, *, channel_axis=-1):

        rgb = torch.maximum(rgb, torch.tensor(1e-6))  

        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.tensor(0))

        return stains
    def combine_stains(self,stains, conv_matrix, *, channel_axis=-1):

        log_rgb = -torch.matmul((stains * -self.log_adjust) , conv_matrix)
        rgb = torch.exp(log_rgb)

        return torch.clamp(rgb, min=0, max=1)
