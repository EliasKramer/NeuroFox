#include "conv_kernel.hpp"

conv_kernel::conv_kernel(int kernel_size)
	:weights(matrix(kernel_size, kernel_size, 0))
{}

void conv_kernel::set_kernel_depth(int depth)
{
	weights.resize(get_kernel_size(), get_kernel_size(), depth);
}

void conv_kernel::set_bias_format(int byte_size)
{
	bias.resize(byte_size, byte_size, 1);
}

matrix& conv_kernel::get_weights()
{
	return weights;
}

const matrix& conv_kernel::get_weights_readonly() const
{
	return weights;
}

matrix& conv_kernel::get_bias()
{
	return bias;
}

const matrix& conv_kernel::get_bias_readonly() const
{
	return bias;
}

size_t conv_kernel::get_kernel_size() const
{
	return weights.get_width();
}