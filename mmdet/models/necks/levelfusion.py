import torch
import torch.nn.functional as F
from torch import nn

from mmcv.runner import BaseModule

from ..builder import NECKS

@NECKS.register_module()
class LevelFusion(BaseModule):
    """"Fuse the multi-level feature map from feature pyramid type backbone.
    
    Args:
        in_channels (List[int]): The channel dimenssions of feature pyramid.
        out_channels (int): The output channel dimenssion of fused feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None. 
    """

    def __init__(self, 
                 in_channels,
                 out_channels: int,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, 1, 1, 0, bias=False)
        self.conv4 = nn.Conv2d(in_channels[3], out_channels, 1, 1, 0, bias=False)

        self.downsample = nn.Conv2d(in_channels[0], in_channels[1], 3, 2, 1, bias=True)
        self.linear_proj = nn.Conv2d(4 * out_channels, out_channels, 1, 1, 0, bias=False)
        

    def forward(self, inputs):
        # channel dimenssions alignment
        feat0 = self.conv1(inputs[0])
        feat1 = self.conv1(inputs[1])
        feat2 = self.conv1(inputs[2])
        feat3 = self.conv1(inputs[3])

        # multi level feature fusion
        feat0 = F.relu(self.downsample(feat0))
        feat2 = F.interpolate(
            feat2, 
            size=feat1.shape[-2:],
            mode='bilinear',
            align_corners=True)
        feat3 = F.interpolate(
            size=feat1.shape[-2:],
            mode='bilinear',
            align_corners=True
        )

        fused_feat = torch.cat((feat0, feat1, feat2, feat3), dim=1)
        
        return self.linear_proj(fused_feat)
