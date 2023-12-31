#encoding:utf-8
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
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        c = 0.044715
        x = ctx.saved_tensors[0]
        tanh_param = sqrt_2_over_pi * (x + c * x**3)
        
        tanh_value = torch.tanh(tanh_param)
        exp_term = torch.exp(-0.5 * (x + c * x**3)**2)
        cdf_term = 0.5 * (1 + torch.erf(x / math.sqrt(2.0)))
        
        derivative = 0.5 * (1 + tanh_value) + \
                    0.5 * x * (1 - tanh_value**2) * (sqrt_2_over_pi) * (1 + c*3 * x**2) * exp_term - \
                    0.5 * x * (1 + tanh_value) * exp_term * cdf_term
        
        return derivative*grad_output

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

