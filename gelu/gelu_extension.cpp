#include <torch/extension.h>

torch::Tensor custom_gelu_cpu(const torch::Tensor& input) {
    return 0.5 * input * (1 + torch::tanh(0.7978845608028654 * (input + 0.044715 * input * input)));
}

PYBIND11_MODULE(custom_gelu_cpu, m) {
    m.def("custom_gelu_cpu", &custom_gelu_cpu, "Custom GELU (CPU) implementation");
}
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
//�ǹ̶�д����m.def("custom_gelu_cpu", &custom_gelu_cpu, "Custom GELU (CPU) implementation")
//�е�custom_gelu_cpu���Զ���ĺ�������&custom_gelu_cpu�Ǻ���ָ�룬
//"Custom GELU (CPU) implementation"�Ǻ�����˵���ĵ���
