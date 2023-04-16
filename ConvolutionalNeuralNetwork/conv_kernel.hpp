#pragma once
#include "matrix.hpp"

class conv_kernel {
private:
	matrix weights;
	float bias = 0;

public:
	conv_kernel(int kernel_size);

	void set_kernel_depth(int depth);
	matrix& get_weights();
	const matrix& get_weights_readonly() const;

	float get_bias();
	void set_bias(float given_bias);

	size_t get_kernel_size() const;

	/// <summary>
	/// sums up all the values in the input matrix multiplied by the kernel
	/// </summary>
	/// <param name="start_x/y">
	///		the top left position in the input matrix, 
	///		where the kernel gets laid over</param>
	/// <param name="kernel_size">
	///		the size of the kernel, 
	///		that will be laid over the matrix</param>
	float lay_kernel_over_matrix(
		const matrix& input_matrix,
		int start_x,
		int start_y);
};