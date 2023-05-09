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

protected:
	e_layer_type_t type;
	//the current matrix of neurons
	matrix activations;

	//the format, that the previous layer has
	matrix input_format;
	//the pointer to the neurons of the previous layer
	const matrix* input = nullptr;

	//the error has the same format as our neurons
	matrix error;
	//the passing error has the same format
	//as the neurons of the previous layer
	matrix* passing_error = nullptr;

	//GPU section
	gpu_matrix* gpu_input = nullptr;
	std::unique_ptr<gpu_matrix> gpu_activations = nullptr;
	std::unique_ptr<gpu_matrix> gpu_error = nullptr;
	gpu_matrix* gpu_passing_error = nullptr;

	bool should_use_gpu();

	virtual void forward_propagation_cpu() = 0;
	virtual	void back_propagation_cpu() = 0;

	virtual void forward_propagation_gpu() = 0;
	virtual void back_propagation_gpu() = 0;

public:
	layer(e_layer_type_t given_layer_type);

	const e_layer_type_t get_layer_type() const;
	const matrix* get_input_p() const;

	void set_input(const matrix* input);
	virtual void set_input_format(const matrix& given_input_format);
	
	void set_previous_layer(layer& previous_layer);

	const matrix& get_activations() const;
	matrix* get_activations_p();

	void set_error_for_last_layer(const matrix& expected);

	//set all weights and biases to that value
	virtual void set_all_parameter(float value) = 0;
	//a random value to the current weights and biases between -value and value
	virtual void apply_noise(float range) = 0;
	//add a random value between range and -range to one weight or bias 
	virtual void mutate(float range) = 0;

	void forward_propagation();
	void back_propagation();

	//the deltas got calculated in the backprop function
	//all the deltas got summed up. now we need to apply the
	//average. this is done by dividing the deltas by the number of inputs
	virtual void apply_deltas(int number_of_inputs) = 0;

	virtual void enable_gpu();
	virtual void disable_gpu();
};