import torch.nn as nn
from orbit.model import BaseBlock, register_model


@register_model()
class NLayerDiscriminator(BaseBlock):
    '''
    PatchGAN 判别器。
    输出不是一个标量，而是一个 N x N 的矩阵，每个点代表对应 Patch 是真还是假。
    '''
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)
