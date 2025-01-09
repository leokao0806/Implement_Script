# import library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from timm.models.layers import trunc_normal_

# check GPU is work then return cuda ; else return cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Block(nn.Module):
    def __init__(self, input_channel, drop_path=0, layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        # Dwconv 7x7, stride 1, padding 3
        self.conv1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=input_channel,
            kernel_size=7,
            groups=input_channel,
            padding=3,
        )
        # LayerNorm
        self.norm1 = LayerNorm(input_channel, eps=1e-6, data_format="channels_first")

        # pwconv 1x1, stride 1
        self.conv2 = nn.Conv2d(
            in_channels=input_channel, out_channels=input_channel * 4, kernel_size=1
        )
        # self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        # GeLU
        self.gelu = nn.GELU()

        # Dwconv 1x1, stride 1
        self.conv3 = nn.Conv2d(
            in_channels=input_channel * 4, out_channels=input_channel, kernel_size=1
        )
        # self.pwconv2 = nn.Linear(dim, 4 * dim)

        # LayerScale
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((input_channel, 1, 1)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        # droppath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = self.drop_path(x)

        x = x + identity

        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # weight default 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # bias default 0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # check data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # normalized_shape
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":  # (batch_size, height, width, channels)
            # Pytorch default layer_norm
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif (
            self.data_format == "channels_first"
        ):  # (batch_size, channels, height, width)
            u = x.mean(1, keepdim=True)
            # square
            s = (x - u).pow(2).mean(1, keepdim=True)
            # std
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            """
            # 假設我們有一個形狀為 (3,) 的一維張量
            x = torch.tensor([1, 2, 3])
            y = x[:, None, None]
            print(x.shape)  # 輸出: torch.Size([3])
            print(y.shape)  # 輸出: torch.Size([3, 1, 1])
            """
            return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    # keep the probability
    keep_prob = 1 - drop_prob
    # (1,) * (x.ndim - 1) means 1,1,1 plus the batch size will be the same shape as x(32,1,1,1)
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    # bernoulli_ is a function that returns a tensor with the same shape as x, with each element being 1 with probability keep_prob and 0 otherwise
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # make sure the tensor is in the same device as x
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class ConvNeXt(nn.Module):

    def __init__(
        self,
        in_channels=3,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        num_classes=1000,
        head_init_scale=1.0,
    ):
        super(ConvNeXt, self).__init__()
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        # stem 4x4 conv
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )  # eps=1e-6 防止除以零的情況
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        input_channel=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        # final norm layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        # Fully connected layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
