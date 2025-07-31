import torch
from torch import nn

class ecgTransForm(nn.Module):
    def __init__(self, configs, hparams):
        super(ecgTransForm, self).__init__()

        filter_sizes = [5, 9, 11, 15, 31, 63]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))
        self.conv4 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[3],
                               stride=configs.stride, bias=False, padding=(filter_sizes[3] // 2))
        self.conv5 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[4],
                               stride=configs.stride, bias=False, padding=(filter_sizes[4] // 2))
        self.conv6 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[5],
                               stride=configs.stride, bias=False, padding=(filter_sizes[5] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)


        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=5)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(hparams["feature_dim"], configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in, return_features=False):
        # 添加维度调整代码
        if x_in.dim() == 2:
            x_in = x_in.unsqueeze(1)  # 把输入数据的维度从 (batch_size, sequence_length) 变为 (batch_size, 1, sequence_length)

        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        x4 = self.conv4(x_in)
        x5 = self.conv5(x_in)
        x6 = self.conv6(x_in)
        x_concat = torch.mean(torch.stack([x1, x2, x3, x4, x5, x6],2), 2)
        
        # x_concat = torch.mean(torch.stack([x1, x2, x3],2), 2)
        
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))
        # print("x_concat shape:", x_concat.shape)

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)
        # print("x shape:", x.shape)

        # Bi-directional Transformer
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x,[2]))
        x = x1+x2

        # original_x = x
        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        x_out_prob = torch.softmax(x_out, dim=1)
        # 如果需要返回特征，则返回 x_flat 和 x_out_prob，original_x测试过效果不好
        if return_features:
            return x_out_prob, x_flat
        else:
            return x_out_prob


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# debug用
# class Configs:
#     input_channels = 1
#     mid_channels = 64

#     final_out_channels = 128
#     stride = 1
#     dropout = 0.3
#     trans_dim = 500
#     num_heads = 5
#     num_classes = 10  

# if __name__ == "__main__":
#     configs = Configs()
#     hparams = {
#         "feature_dim": configs.final_out_channels,  # 对应 AdaptiveAvgPool1d(1) 输出后展平的维度
#     }

#     model = ecgTransForm(configs, hparams)

#     # 构造一个假数据输入 (batch_size=8, signal_length=4000)
#     dummy_input = torch.randn(8, 3984)

#     # 前向传播测试
#     output = model(dummy_input)

#     print("Output shape:", output.shape)  # 期望输出为 (8, num_classes)
