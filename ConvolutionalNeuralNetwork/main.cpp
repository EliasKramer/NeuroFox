#include <iostream>
#include <vector>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"
#include "gpu_matrix.cuh"

int main()
{
    std::vector<float> kernel_data = {
        1, 2,
        3, 4
    };
    matrix kernel(2, 2, 1, kernel_data);

    std::vector<std::unique_ptr<gpu_matrix>> gpu_kernel_weights;
    gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel, true));

    return 0;
}