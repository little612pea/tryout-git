#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <cmath>
#include <vector>
namespace py = pybind11;

torch::Tensor gelu_forward(const torch::Tensor& input) {
    return 0.5 * input * (1 + tanh(M_2_SQRTPI * M_SQRT1_2 * (input + 0.044715 * input * input * input)));
}
torch::Tensor gelu_backward(
    const torch::Tensor& grad,
    const torch::Tensor& self) {
  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  torch::Tensor cdf = (1.0 + (self * M_SQRT1_2).erf_()).mul_(0.5);
  torch::Tensor pdf = (-0.5 * self * self).exp_();
  return cdf.addcmul_(self, pdf, kAlpha).mul_(grad);
}

PYBIND11_MODULE(gelu_cpp, m) {
  m.def("forward", &gelu_forward, "gelu forward");
  m.def("backward", &gelu_backward, "gelu backward");
}