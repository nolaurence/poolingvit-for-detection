import enum
import warnings
import torch
import math
import numpy as np

from typing import Optional
from mmcv.cnn.bricks.conv import build_conv_layer

from torch import nn
from torch.nn import functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import AvgPool2d, MaxPool2d
from torch.nn.modules.utils import _pair as to_2tuple

from ..builder import BACKBONES
from ..utils.transformer import (PatchEmbed, nchw_to_nlc, nlc_to_nchw)
from ...utils import get_root_logger


class MixFFN(BaseModule):
    """An implementation of MixFFN of PVT.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, activate, drop, fc2, drop]
        if use_conv:
            layers.insert(1, dw_conv)
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class IdentityPooling(BaseModule):
    """The pooling operation adding to pooling attention's identity connection

    Args:
        pooling_kernel_size (int): The size of kernel for pooling layer.
            Default: 2.
        pooling_ratio (int): The pooling ratio for pooling layer.
            Default: 2.
        pooling_type (string): to determine the type of pooling operation.
        backbone_type (string): Identify the type of backbone to determine whether use
            dimension transform. Default: 'ResNet'.
    """

    def __init__(self,
                 pooling_kernel_size: int = 2,
                 pooling_ratio: int = 2,
                 pooling_type: str = "avg",
                 backbone_type: str = "ResNet"):
        super(IdentityPooling, self).__init__()

        self.backbone_type = backbone_type

        assert backbone_type in [
            "ResNet", "ViT"], "backbone time should be ResNet or ViT"

        if pooling_type == "avg":
            self.pooling_layer = AvgPool2d(pooling_kernel_size, pooling_ratio)
        elif pooling_type == "max":
            self.pooling_layer = MaxPool2d(pooling_kernel_size, pooling_ratio)

    def forward(self, x, hw_shape=None):
        if self.backbone_type == "ResNet":

            # In ResNet type, the main shape of feature map is [N, C, H, W]
            # In ViT type, the shape of feature map is [N, L, C], where L = H * W.
            return self.pooling_layer(x)

        elif self.backbone_type == "ViT":
            assert hw_shape is not None, "If backbone type is ViT, hw_shape shouldn't be ommited."
            out = nlc_to_nchw(x, hw_shape)
            out = self.pooling_layer(out)
            return nchw_to_nlc(out)
        else:
            raise "Backbone type should be ResNet or ViT."


class AdditiveSelfAttention(BaseModule):
    '''An implementation of Additive self attention

    The difference between vanilla multi head self-attention and Additive self attention is:
        1. remove the Q*K^T calculation for attention weight.
        2. use additive calculation for global query and global

    Args: 
        dim (int): The dimension of the input sequence.
        decode_dim (int): Similar to hidden dimension in MultiheadAttention.
    '''

    def __init__(self, dim: int = 3, decode_dim: int = 16):
        '''
        dim: the hidden dimension of input embedding features
        decode_dim: unknown by far (21点06分 2021年12月14日)
        TODO: Figure out the function of the decode_dim
        '''
        super(AdditiveSelfAttention, self).__init__()

        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)

        query = query
        key = key

        # b - batch size, n - length of sequence, d - dimension of embeddings
        b, n, d = query.size()

        mask_value = -torch.finfo(x.dtype).max
        # mask = rearrange(mask, 'b n -> b () n')  # should be the same as mask = mask.view(b, -1, n)

        # 兼容无mask输入的情况
        if mask is None:
            mask = torch.ones(b, n).bool().to(self.weight_alpha.device)
        # mask = mask.view(b, -1, n)
        mask = mask.unsqueeze(dim=-1)

        # calculate the global query vector
        # query * W_{alpha} is the way to produce
        # alpha_weight = (torch.mul(query, self.weight_alpha) *
        #                 self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = (query * self.weight_alpha *
                        self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim=-1)
        # [b, n, d] -> [b, d] sum all querys to produce global vector
        global_query = (query * alpha_weight).sum(dim=1)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = global_query.unsqueeze(dim=1).repeat(1, n, 1)
        p = repeat_global_query * key
        beta_weight = (p * self.weight_beta *
                       self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim=-1)
        # dim change is the same as global query
        global_key = (p * beta_weight).sum(dim=1)

        # key-value interaction
        repeat_global_key = global_key.unsqueeze(dim=1).repeat(1, n, 1)
        u = repeat_global_key * value
        key_value_interaction_output = self.weight_r(u)

        return key_value_interaction_output + query


class PoolingAttention(MultiheadAttention):
    '''
    Implementation of Pooling Attention.

    This module is modified from MultiheadAttention which is a module from mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        pooling_kernel_size (int): The size of kernel for pooling layer.
            Default: 2.
        pooling_ratio (int): The pooling ratio for pooling layer.
            Default: 2.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    '''

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 dropout_layer: object = None,
                 batch_first: bool = True,
                 qkv_bias: bool = True,
                 pooling_kernel_size: int = 2,
                 pooling_ratio: int = 2,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg)

        self.pooling_ratio = pooling_ratio

        if pooling_ratio > 1:
            self.sequence_reduction = nn.AvgPool2d(
                pooling_kernel_size, pooling_ratio)

    def forward(self, x, hw_shape):
        x_query = x
        if self.pooling_ratio > 1:
            x_key_value = nlc_to_nchw(x, hw_shape)
            x_key_value = self.sequence_reduction(x_key_value)
            x_key_value = nchw_to_nlc(x_key_value)
        else:
            x_key_value = x

        # in "torch.nn.MultiheadAttention" is [num_query, batch, embed_dims], here we use batch first, which is
        # [batch, num_query, embed_dims], so here is a transpose.
        if self.batch_first:
            x_query = x_query.transpose(0, 1)
            x_key_value = x_key_value.transpose(0, 1)

        # Point: torch.nn.MultiheadAttention returns: [output, attention weights]
        output = self.attn(query=x_query, key=x_key_value,
                           value=x_key_value)[0]

        if self.batch_first:
            output = output.transpose(0, 1)

        return output


class LearnedPositionEmbedding(BaseModule):
    """An implmentation of the learned position embedding in Pooling ViT.

    Args:
        pos_shape (int): The shape of the learned position embedding.
        pos_dim (int): The dimension of the learned position embedding.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.0.
    """

    def __init__(self, pos_shape: int, pos_dim: int, drop_rate: float = 0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pos_shape, int):
            pos_shape = to_2tuple(pos_shape)
        elif isinstance(pos_shape, tuple):
            if len(pos_shape) == 1:
                pos_shape = to_2tuple(pos_shape[0])
            assert len(pos_shape) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pos_shape)}'
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
        """Resize pos_embed weights.

        Resize pos_embed using bilinear interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bilinear'``.

        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C].
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight,
                                         2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)


class PoolingViTEncoderLayer(BaseModule):
    """Transformer style base block.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): number of heads in Vanilla Self Attention and Pooling Attention.
        feedforward_channels (int): The hidden dimension of FFNs.
        drop_rate (int): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer. Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if true. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type="GELU").
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        use_pooling (bool): If true, it represents the layer is the first layer of one stage.
            Default: false.
        pooling_kernel_size (int, optional): The size of kernel for pooling layer.
            Default: 2.
        pooling_ratio (int, optional): The pooling ratio for pooling layer.
            Default: 2.
        use_conv_ffn (bool): If true, use Convolutional FNN to replace FFN.
            Default: false.
        use_additifve_attn (bool): The flag to determine whether use additive attention.
            Default: True
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type="GELU"),
                 norm_cfg: dict = dict(type='LN'),
                 use_pooling: bool = False,
                 pooling_kernel_size: Optional[int] = 2,
                 pooling_ratio: Optional[int] = 2,
                 use_conv_ffn: bool = False,
                 use_additive_attn: bool = True,
                 init_cfg: Optional[float] = None):
        super(PoolingViTEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.use_pooling = use_pooling

        # The return[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        if use_pooling:
            self.attn = PoolingAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias,
                pooling_kernel_size=pooling_kernel_size,
                pooling_ratio=pooling_ratio)
        else:
            if use_additive_attn:
                self.attn = AdditiveSelfAttention(
                    dim=embed_dims, decode_dim=embed_dims)
            else:
                self.attn = MultiheadAttention(
                    embed_dims, num_heads, attn_drop_rate, drop_rate)

        # The return[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        if self.use_pooling:
            x1 = self.attn(self.norm1(x), hw_shape)
            x = x1 + x
        else:
            x1 = self.attn(self.norm1(x), key=self.norm1(x),
                           value=self.norm1(x))
            x = x1 + x

        x = self.ffn(self.norm2(x), hw_shape, identity=x)

        return x


class PoolingViTBottleneckBlock(BaseModule):
    """Implementation of ResNet style bottleneck block.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of mid attention layer.
            Default: 4.
        stride (int): stride of the block. Default: 1.
        dilation (int): dilation of convolution. Default: 1.
        downsample (bool): identify if use downsample in identity connection.
            Default: false.
        use_additive_attention (bool): The flag to determine whether use additive attention.
            Default: True
        num_heads (int): Tne number of heads for multi head attention layer.
            Default: 8.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        init_cfg (dict): dictionary for initialization.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int = 4,
                 stride: int = 1,
                 use_additive_attention: bool = True,
                 pooling_kernel_size: Optional[int] = 2,
                 pooling_ratio: Optional[int] = 2,
                 num_heads: Optional[int] = 8,
                 norm_cfg=dict(type='BN'),
                 conv_cfg: Optional[dict] = None,
                 init_cfg=None):
        super(PoolingViTBottleneckBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.use_additive_attention = use_additive_attention
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_ratio = pooling_ratio
        self.num_heads = num_heads
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            dict(type='LN'), self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, self.out_channels, postfix=3)

        self.in_conv = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if use_additive_attention:
            self.attention = AdditiveSelfAttention(
                self.mid_channels, self.mid_channels)
        else:
            self.attention = PoolingAttention(
                self.mid_channels, self.num_heads)
        self.add_module(self.norm2_name, norm2)

        self.out_conv = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x, hw_shape):
        identity = x

        x = self.in_conv(x)
        x = self.norm1(x)
        x = self.relu(x)

        mid = nchw_to_nlc(x)
        if self.use_additive_attention:
            mid = self.attention(mid)
        else:
            mid = self.attention(mid, hw_shape)
        mid = self.norm2(mid)
        out = nlc_to_nchw(mid, hw_shape)

        out = self.out_conv(out)
        self.norm3(out)

        out = out + identity
        return self.relu(out)


@BACKBONES.register_module()
class PoolingVisionTransformer(BaseModule):
    """Pooling Vision Transformer (PoolingVit)

    Implementation of `Pooling Vision Transformer`, which has pooling 
    attention and additive attention.

    Args:
        pretrained_img_size (int): The size of input image when pretrain. 
            Defauls: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stage (int): Number of stages. Default: 4.
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        pooling_ratios (Sequence[int]): The spatial pooling rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        pooling_kernel_size (Sequence[int]): The size of kernel for pooling layer.
            Default: 2.
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_pooling (bool): Determine whether use pooling attention.
            Default: true.
        use_additive_attn (bool): Determine whether use addtive attention.
            Default: false.
        block_type: (str): Determine the type of basic block. pooling vit has encoder layer and bottleneck.
            Default: 'encoder'.
        backbone_mode: (str): Determine the mode of backbone depth.
            Default: 'small'.
        use_learned_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    vit_architecture_settings = {
        'tiny': (2, 2, 2, 2),
        'small': (3, 3, 6, 3),
        'medium': (3, 3, 18, 3),
        'large': (3, 8, 27, 3)
    }

    bottleneck_type_archiecture_settings = {
        'tiny': (2, 2, 2, 2),
        'small': (3, 4, 6, 3),
        'medium': (3, 4, 23, 3),
        'large': (3, 8, 36, 3)
    }

    def __init__(self,
                 pretrain_img_size: int = 224,
                 in_channels: int = 3,
                 embed_dims: int = 64,
                 num_stages: int = 4,
                 num_heads: list = [1, 2, 5, 8],
                 patch_sizes: list = [4, 2, 2, 2],
                 strides: list = [4, 2, 2, 2],
                 paddings: list = [0, 0, 0, 0],
                 pooling_ratios: list = [8, 4, 2, 1],
                 pooling_kernel_sizes: list = [8, 4, 2, 1],
                 out_indices: tuple = (0, 1, 2, 3),
                 mlp_ratios: list = [8, 8, 4, 4],
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_learned_pos_embed: bool = True,
                 use_pooling: bool = True,
                 use_additive_attn: bool = False,
                 block_type: str = 'encoder',
                 backbone_mode: str = 'small',
                 norm_after_stage: bool = False,
                 use_conv_ffn: bool = False,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 conv_cfg: dict = None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims

        self.num_stages = num_stages
        assert backbone_mode in self.vit_architecture_settings.keys(), \
            'wrong backbone type, it should be tiny, small, medium or large'
        assert block_type in ['bottleneck', 'encoder'], \
            "only bottleneck and encoder block type supported"
        if block_type == 'encoder':
            self.num_layers = self.vit_architecture_settings[backbone_mode]
        elif block_type == 'bottleneck':
            self.num_layers = self.bottleneck_type_archiecture_settings[backbone_mode]
        # self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides

        self.pooling_ratio = pooling_ratios
        self.pooling_kernel_size = pooling_kernel_sizes
        assert num_stages == len(self.num_layers) == len(num_heads) \
            == len(patch_sizes) == len(strides) == len(pooling_ratios) == len(pooling_kernel_sizes)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained

        # init of backbone
        drop_path_rate_list = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(self.num_layers))]

        cur = 0
        self.layers = ModuleList()

        if block_type == 'encoder':
            for i, num_layer in enumerate(self.num_layers):
                embed_dims_i = embed_dims * num_heads[i]
                patch_embed = PatchEmbed(
                    in_channels=in_channels,
                    embed_dims=embed_dims_i,
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=True,
                    norm_cfg=norm_cfg)

                layers = ModuleList()
                if use_learned_pos_embed:
                    pos_shape = pretrain_img_size // np.prod(
                        patch_sizes[:i + 1])
                    pos_embed = LearnedPositionEmbedding(
                        pos_shape, embed_dims_i, drop_rate)
                    layers.append(pos_embed)

                layers.extend([
                    PoolingViTEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[i],
                        feedforward_channels=mlp_ratios[i] * embed_dims_i,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=drop_path_rate_list[cur + idx],
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        use_pooling=use_pooling,
                        use_additive_attn=use_additive_attn,
                        pooling_kernel_size=pooling_kernel_sizes[i],
                        pooling_ratio=pooling_ratios[i],
                        use_conv_ffn=use_conv_ffn) for idx in range(num_layer)
                ])
                in_channels = embed_dims_i
                # The return[0] of build_norm_layer is norm name.
                if norm_after_stage:
                    norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
                else:
                    norm = nn.Identity()
                self.layers.append(ModuleList([patch_embed, layers, norm]))
                cur += num_layer

        elif block_type == "bottleneck":
            for i, num_layer in enumerate(self.num_layers):
                embed_dims_i = embed_dims * num_heads[i]
                patch_embed = PatchEmbed(
                    in_channels=in_channels,
                    embed_dims=embed_dims_i,
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=True,
                    norm_cfg=norm_cfg)

                layers = ModuleList()
                if use_learned_pos_embed:
                    pos_shape = pretrain_img_size // np.prod(
                        patch_sizes[:i + 1])
                    pos_embed = LearnedPositionEmbedding(
                        pos_shape=pos_shape,
                        pos_dim=embed_dims_i,
                        drop_rate=drop_rate)
                    layers.append(pos_embed)
                layers.extend([
                    PoolingViTBottleneckBlock(
                        in_channels=mlp_ratios[i] * embed_dims_i,
                        out_channels=mlp_ratios[i] * embed_dims_i,
                        expansion=mlp_ratios[i],
                        use_additive_attention=use_additive_attn,
                        pooling_kernel_size=pooling_kernel_sizes[i],
                        pooling_ratio=pooling_ratios[i],
                        num_heads=num_heads[i],
                        norm_cfg=norm_cfg,
                        conv_cfg=conv_cfg) for idx in range(num_layer)
                ])
                in_channels = embed_dims_i
                if norm_after_stage:
                    norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
                else:
                    norm = nn.Identity()
                self.layers.append(ModuleList([patch_embed, layers, norm]))
                cur += num_layer

        self.block_type = block_type

    def init_weight(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, LearnedPositionEmbedding):
                    m.init_weights()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []

        if self.block_type == 'encoder':
            for i, layer in enumerate(self.layers):
                x, hw_shape = layer[0](x)

                for block in layer[1]:
                    x = block(x, hw_shape)
                x = layer[2](x)
                x = nlc_to_nchw(x, hw_shape)

                if i in self.out_indices:
                    outs.append(x)

        if self.block_type == 'bottleneck':
            for i, layer in enumerate(self.layers):
                x, hw_shape = layer[0](x)
                x = nlc_to_nchw(x, hw_shape)

                for block in layer[1]:
                    x = block(x, hw_shape)
                x = layer[2](x)

                if i in self.out_indices:
                    outs.append(x)

        return outs[-1]
