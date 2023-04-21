#pragma once
#include "matrix.hpp"

class conv_kernel {
private:
	matrix weights;
	matrix bias;

public:
	conv_kernel(int kernel_size);

	void set_kernel_depth(int depth);
	void set_bias_format(int size);

	matrix& get_weights();
	const matrix& get_weights_readonly() const;

	matrix& get_bias();
	const matrix& get_bias_readonly() const;

	size_t get_kernel_size() const;
};