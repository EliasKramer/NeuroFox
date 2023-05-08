#include <iostream>
#include <vector>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"

int main()
{
    std::vector<float> kernel_data = {
        1, 2,
        3, 4
    };
    matrix kernel(kernel_data, 2, 2, 1);

    std::vector<std::unique_ptr<gpu_memory<float>>> gpu_kernel_weights;
    gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel));

    return 0;
}