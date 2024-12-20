import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class FeatureDistillationLoss(nn.Module):
    def __init__(self, extractor):
        super(FeatureDistillationLoss, self).__init__()
        self.extractor = extractor  
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def forward(self, input, target):

        input, target = self.transform(input), self.transform(target)
        with torch.no_grad():
            input_features = self.extractor(input).squeeze()  
            target_features = self.extractor(target).squeeze()  

        cosine_similarity = F.cosine_similarity(input_features, target_features, dim=-1)  
        loss = 1 - cosine_similarity.mean()  

        
        return loss