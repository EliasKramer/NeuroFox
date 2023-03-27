#pragma once
#include "matrix.hpp"

struct _neural_kernel {
	matrix weights;
	matrix biases;
	matrix output;
} typedef neural_kernel;