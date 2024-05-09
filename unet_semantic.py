import torch
import torch.nn as nn
import torch.nn.init as init

class UNet(nn.Module):
    def __init__(self, n_class, dropout=0.06742106126314273):
        super().__init__()
        self.dropout = dropout
        # Helper function for creating a block
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        # Encoder
        self.e11 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = conv_block(512, 1024)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e61 = conv_block(1024, 2048)
        
        # Decoder with Upsampling
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d11 = conv_block(2048 + 1024, 1024)  # The channel size includes the concatenated features

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d21 = conv_block(1024 + 512, 512)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d31 = conv_block(512 + 256, 256)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d41 = conv_block(256 + 128, 128)

        self.upconv5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d51 = conv_block(128 + 64, 64) 

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

        # Apply He initialization to all convolutional layers
        self._initialize_weights()

    # He weight initialization reduces the effect of exploding/vanishing gradients
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e11 = self.e11(x)
        pool1 = self.pool1(e11)

        e21 = self.e21(pool1)
        pool2 = self.pool2(e21)

        e31 = self.e31(pool2)
        pool3 = self.pool3(e31)

        e41 = self.e41(pool3)
        pool4 = self.pool4(e41)

        e51 = self.e51(pool4)
        pool5 = self.pool5(e51)

        e61 = self.e61(pool5)

        # Decoder with Upsampling
        up1 = self.upconv1(e61)
        up1 = torch.cat([up1, e51], dim=1)
        d11 = self.d11(up1)

        up2 = self.upconv2(d11)
        up2 = torch.cat([up2, e41], dim=1)
        d21 = self.d21(up2)

        up3 = self.upconv3(d21)
        up3 = torch.cat([up3, e31], dim=1)
        d31 = self.d31(up3)

        up4 = self.upconv4(d31)
        up4 = torch.cat([up4, e21], dim=1)
        d41 = self.d41(up4)

        up5 = self.upconv5(d41)
        up5 = torch.cat([up5, e11], dim=1)
        d51 = self.d51(up5)

        # Output layer
        out = self.outconv(d51)
        
        bottleneck_features = e61
        return out, bottleneck_features