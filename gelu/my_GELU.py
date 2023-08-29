import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * input * (1.0 + torch.tanh((math.sqrt(2.0 / math.pi) * (input + 0.044715 * input**3))))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        exp_term = torch.exp(-0.5 * input**2)
        cdf_term = 0.5 * (1 + torch.erf(input / math.sqrt(2.0)))
        gradient = 0.5 * (1.0 + torch.tanh((math.sqrt(2.0 / math.pi) * (input + 0.044715 * input**3)))) + \
                   0.5 * input * (1 - torch.tanh((math.sqrt(2.0 / math.pi) * (input + 0.044715 * input**3)))**2) * \
                   (math.sqrt(2.0 / math.pi) * (1 + 0.044715 * input**2)) * exp_term - \
                   0.5 * input * (1.0 + torch.tanh((math.sqrt(2.0 / math.pi) * (input + 0.044715 * input**3)))) * exp_term * cdf_term
        return grad_output * gradient

class MyGELU(nn.Module):
    def forward(self, input):
        return MyGELUFunction.apply(input)
    def backward(self, input):
        return MyGELUFunction.apply(input)
