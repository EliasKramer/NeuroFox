#pragma once
#include "matrix.hpp"
enum _activation_function
{
	sigmoid,
	relu
} typedef activation_function;

struct _neural_kernel {
	matrix weights;
	matrix biases;
	matrix output;
} typedef neural_kernel;