#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "layer.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class fully_connected_layer : public layer {
private:
	matrix weights;
	matrix biases;

	matrix weight_deltas;
	matrix bias_deltas;

	e_activation_t activation_fn;

	//GPU Section
	std::unique_ptr<gpu_matrix> gpu_weights = nullptr;
	std::unique_ptr<gpu_matrix> gpu_biases = nullptr;

	float get_weight_at(int input_layer_idx, int current_activation_idx) const;
	void set_weight_at(int input_layer_idx, int current_activation_idx, float value);

	float get_weight_delta_at(int input_layer_idx, int current_activation_idx) const;
	void set_weight_delta_at(int input_layer_idx, int current_activation_idx, float value);

public:

	fully_connected_layer(
		int number_of_neurons,
		e_activation_t activation_function
	);
	
	fully_connected_layer(
		const matrix& activation_format,
		e_activation_t activation_function
	);

	void set_input_format(const matrix& input_format) override;

	const matrix& get_weights() const;
	const matrix& get_biases() const;
	matrix& get_weights_ref();
	matrix& get_biases_ref();

	//set all weights and biases to that value
	void set_all_parameter(float value) override;
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range) override;
	//add a random value between range and -range to one weight or bias 
	void mutate(float range) override;

	void forward_propagation_cpu(const matrix& input) override;
	void back_propagation_cpu(const matrix& input, matrix* passing_error) override;

	void forward_propagation_gpu(const gpu_matrix& input) override;
	void back_propagation_gpu(const gpu_matrix& input, gpu_matrix* passing_error) override;

	void apply_deltas(int number_of_inputs) override;

	void enable_gpu() override;
	void disable_gpu() override;
};