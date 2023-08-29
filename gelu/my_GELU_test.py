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

# Validate the custom GELU implementation
A = torch.randn(100)
B = A.clone()
A.requires_grad = True
B.requires_grad = True
c = torch.randn(100)

# Using torch.nn.functional.gelu
a = F.gelu(A, approximate="tanh")

# Using custom GELU
my_gelu = MyGELU()
b = my_gelu(B)

loss_func = nn.MSELoss()
loss1 = loss_func(a, c)
loss2 = loss_func(b, c)

print("Loss using torch.nn.functional.gelu:", loss1.item())
print("Loss using custom GELU:", loss2.item())

loss1.backward()
loss2.backward()

gradA = A.grad
gradB = B.grad

err = loss_func(gradA, gradB)
print("Gradient error between torch.nn.functional.gelu and custom GELU:", err.item())

# Print gradient matrices
print("Gradient matrix for torch.nn.functional.gelu:\n", gradA)
print("Gradient matrix for custom GELU:\n", gradB)