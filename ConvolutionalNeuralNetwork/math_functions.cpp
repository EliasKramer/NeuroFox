#include "math_functions.hpp"
#include <cmath>

float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

float relu(float x)
{
	return x > 0 ? x : 0;
}

float sigmoid_derivative(float x)
{
	float sig = sigmoid(x);
	return sig * (1.0f - sig);
}

float relu_derivative(float x)
{
	return x > 0 ? 1.0f : 0;
}
