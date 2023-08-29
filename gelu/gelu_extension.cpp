#include <torch/extension.h>

torch::Tensor custom_gelu_cpu(const torch::Tensor& input) {
    return 0.5 * input * (1 + torch::tanh(0.7978845608028654 * (input + 0.044715 * input * input)));
}

PYBIND11_MODULE(custom_gelu_cpu, m) {
    m.def("custom_gelu_cpu", &custom_gelu_cpu, "Custom GELU (CPU) implementation");
}
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
//是固定写法，m.def("custom_gelu_cpu", &custom_gelu_cpu, "Custom GELU (CPU) implementation")
//中的custom_gelu_cpu是自定义的函数名，&custom_gelu_cpu是函数指针，
//"Custom GELU (CPU) implementation"是函数的说明文档。
