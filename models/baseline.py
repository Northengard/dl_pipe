import torch
from torch import nn

_poolings = {
    'avg': nn.AvgPool2d,
    'max': nn.MaxPool2d,
}


class ConvBnReluPool(nn.Module):

    def __init__(self, in_channels, out_channels, conv_params, pool_type, pool_params):
        """
        Convolution -> ReLu -> BN -> Pooling block as one module
        :param pool_type: str, 'avg' or 'max' are available
        :param pool_params:
        :param conv_params:
        """
        super(ConvBnReluPool, self).__init__()
        cksize, cstride, cpad = conv_params['kernel_size'], conv_params['stride'], conv_params['padding']
        pksize, pstride, ppad = pool_params['kernel_size'], pool_params['stride'], pool_params['padding']

        self.block = nn.Sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=cksize, stride=cstride, padding=cpad),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(),
                                     _poolings[pool_type](kernel_size=pksize, stride=pstride, padding=ppad)])

    def forward(self, x):
        x = self.block(x)
        return x


class ForwardRegression(nn.Module):
    def __init__(self, config):
        """
        Base line model.
        Simple convolutional network
        with N blocks: Convolution-->ReLu-->BatchNorm-->Pooling
        Transforms image to logits K-vector where K is number of classes
        :param config: dictionary with main parameters. See configs/ directory for an example
        """
        super(ForwardRegression, self).__init__()

        self._fconv_out = config['first_conv']['out_channels']
        self._fconv_ksize = config['first_conv']['kernel_size']
        self._fconv_stride = config['first_conv']['stride']
        self._fconv_pad = config['first_conv']['padding']

        self.num_blocks = config['num_blocks']
        self.num_classes = config['num_classes']
        self.pool_type = config['pooling']['type']
        self.mul_rate = config['channels_multiplier']
        self.relu = nn.ReLU()

        self.preprocessing = list()
        self.preprocessing.append(nn.Conv2d(in_channels=config['start_channels'],
                                            out_channels=self._fconv_out,
                                            kernel_size=self._fconv_ksize,
                                            stride=self._fconv_stride,
                                            padding=self._fconv_pad))
        self.preprocessing.append(self.relu)
        self.preprocessing.append(ConvBnReluPool(in_channels=self._fconv_out,
                                                 out_channels=self._fconv_out,
                                                 conv_params=config['base_conv'],
                                                 pool_type=self.pool_type,
                                                 pool_params=config['pooling']))
        self.preprocessing = nn.Sequential(*self.preprocessing)

        self.main_blocks = list()
        block_out_channels = self._fconv_out
        for idx in range(self.num_blocks):
            block_in_channels = block_out_channels
            block_out_channels *= self.mul_rate
            self.main_blocks.append(ConvBnReluPool(in_channels=block_in_channels,
                                                   out_channels=block_out_channels,
                                                   conv_params=config['base_conv'],
                                                   pool_type=self.pool_type,
                                                   pool_params=config['pooling']))
        self.main_blocks = nn.Sequential(*self.main_blocks)
        self.pre_fc = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.pre_fc = torch.flatten
        self.fc1 = nn.Linear(in_features=block_out_channels, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=self.num_classes)

    def forward(self, x):
        x = self.preprocessing(x)
        x = self.main_blocks(x)
        x = self.pre_fc(x)  # (x, start_dim=1)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
