import torch.nn as nn

class Residual(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels, out_channels, use_one_d:bool, stride) -> None:
        super().__init__()
        self.convd_layer_a = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.convd_layer_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.convd_layer_c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride) if use_one_d else None
        self.batchnorm_layer_a = nn.BatchNorm2d(out_channels)
        self.batchnorm_layer_b = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        out_tensor = nn.functional.relu(self.batchnorm_layer_a(self.convd_layer_a(input_tensor)))
        out_tensor = self.batchnorm_layer_b(self.convd_layer_b(out_tensor))
        input_tensor = self.convd_layer_c(input_tensor) if self.convd_layer_c else input_tensor
        out_tensor += input_tensor
        return nn.functional.relu(out_tensor)