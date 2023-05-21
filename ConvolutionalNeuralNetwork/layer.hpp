#pragma once
#include "matrix.hpp"
#include "gpu_matrix.cuh"
#include "gpu_math.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef enum _layer_type {
	convolution,
	pooling,
	fully_connected,
	NO_TYPE
} e_layer_type_t;

class layer {

private:
	void valid_input_check_cpu(const matrix& input) const;
	void valid_passing_error_check_cpu(const matrix* passing_error) const;

	void valid_input_check_gpu(const gpu_matrix& input) const;
	void valid_passing_error_check_gpu(const gpu_matrix* passing_error) const;
protected:
	e_layer_type_t type;

	//the current matrix of neurons
	matrix activations;
	//the current error - the same format as the activations
	matrix error;

	matrix input_format;

	std::unique_ptr<gpu_matrix> gpu_activations = nullptr;
	std::unique_ptr<gpu_matrix> gpu_error = nullptr;

public:
	layer(e_layer_type_t given_layer_type);
	layer(matrix activation_format, e_layer_type_t given_layer_type);

	const e_layer_type_t get_layer_type() const;

	virtual void set_input_format(const matrix& given_input_format);
	
	const matrix& get_activations() const;
	matrix* get_activations_p();

	const matrix& get_error() const;
	matrix* get_error_p();

	//TODO - make this separate
	void set_error_for_last_layer_cpu(const matrix& expected);

	//set all weights and biases to that value
	virtual void set_all_parameter(float value) = 0;
	//a random value to the current weights and biases between -value and value
	virtual void apply_noise(float range) = 0;
	//add a random value between range and -range to one weight or bias 
	virtual void mutate(float range) = 0;

	virtual void forward_propagation_cpu(const matrix& input);
	virtual	void back_propagation_cpu(const matrix& input, matrix* passing_error);

	virtual void forward_propagation_gpu(const gpu_matrix& input);
	virtual void back_propagation_gpu(const gpu_matrix& input, gpu_matrix* passing_error);

	//the deltas got calculated in the backprop function
	//all the deltas got summed up. now we need to apply the
	//average. this is done by dividing the deltas by the number of inputs
	virtual void apply_deltas(size_t training_data_count, float learning_rate) = 0;

	virtual void enable_gpu();
	virtual void disable_gpu();
};