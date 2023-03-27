#pragma once
#include "matrix.hpp"

struct _neural_kernel {
	matrix weights;
	float biase;
	matrix output;
} typedef neural_kernel;