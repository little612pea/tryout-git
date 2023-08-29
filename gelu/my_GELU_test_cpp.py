#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import gelu_cpp

class MyGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_cpp.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return gelu_cpp.backward(grad_output,x)

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

