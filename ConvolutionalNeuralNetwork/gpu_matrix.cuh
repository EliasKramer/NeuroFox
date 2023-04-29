#pragma once
#include <stdexcept>
#include "matrix.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float* copy_to_gpu(const matrix& m);
cudaError_t gpu_add_matrices(const float* gpu_matrix_a, const float* gpu_matrix_b, float* gpu_matrix_result, unsigned int size);
