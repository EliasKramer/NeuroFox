#include "math_functions.hpp"

float sigmoid_fn(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

float relu_fn(float x)
{
	return x > 0 ? x : 0;
}