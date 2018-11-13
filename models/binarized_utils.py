import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function, gradcheck
import numpy as np

def Binarize(tensor, quantization_model='deterministic'):
    if quantization_model == 'deterministic':
        return tensor.sign()
    elif quantization_model == 'stochastic':
        """
        xb: 1 if p = sigma(x)
           -1 if 1-p
        where sigma(x) "is hard sigmoid" function:
            sigma(x) = clip((x+1)/2, 0,1)

        tensor.add_(1).div_(2) => at x=0, output=0.5
        torch.rand(tensor.size()).add(-0.5) => output_1, #隨機產生[0,1]之間的值並減0.5
        if rand < 0.5 => ouput + output_1 < 0.5 => output_2
           rand >= 0.5 => output_2 >= 0.5

        output_2.clamp_(0,1).round() => output_3
        if rand < 0.5 => output_3=0
           rand >= 0.5 => output_3=1

        output_3.mul(2).add(-1) => output_4
        此為將[0,1] -> [-1,1]

        if x=-1 => output=0 => max(ouput_1) < 0.5 => output_2 < 0.5 => output_3 = 0 => 最後輸出 -1
        """
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)

class BinarizeLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinarizeLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:  # 28*28
            #  Any changes on x.data wouldn’t be tracked by autograd
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = F.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinarizeConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

if __name__ == "__main__":
    '''
    x = torch.randn(2,2, requires_grad=True)
    w = torch.randn(2,2, requires_grad=True)
    grad_output = torch.randn(2,2)
    bin_x = x.sign()
    bin_w = w.sign()
    out = bin_x.matmul(bin_w.t())
    out.backward(grad_output)
    print("raw input x: \n{}".format(x))
    print("raw input w: \n{}".format(w))
    print("output: \n{}".format(out))
    print("grad output: \n{}".format(grad_output))
    print("grad_input_x: \n{}".format(x.grad)) # x.grad=0
    print("grad_input_w: \n{}".format(w.grad))
    print("="*50)
    '''

    binlinear = BinarizeLinear(2, 2, bias=False)
    x = torch.randn(2,2, requires_grad=True)
    test = gradcheck(binlinear, (x,), eps=1e-3, atol=1e-4)
    print("Gradient check: ", test)
