#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "layer.hpp"

class fully_connected_layer : public layer {
private:
	matrix weights;
	matrix biases;

	matrix weight_deltas;
	matrix bias_deltas;

	e_activation_t activation_fn;

	float get_weight_at(int input_layer_idx, int current_activation_idx) const;
	void set_weight_at(int input_layer_idx, int current_activation_idx, float value);

	float get_weight_delta_at(int input_layer_idx, int current_activation_idx) const;
	void set_weight_delta_at(int input_layer_idx, int current_activation_idx, float value);

	float get_error_at(int input_layer_idx) const;
	void set_error_at(int input_layer_idx, float value);

public:

	/// <param name="given_input">the pointer where the input will be read</param>
	/// <param name="input_format">
	/// the first layer has usually a nullptr as input
	/// in order to set the right weights and biases, the input format is needed
	/// a format is an empty matrix with the right dimensions
	/// </param>
	/// <param name="number_of_neurons">number of neurons this layer will have</param>
	/// <param name="activation_function">the activation function that will be used</param>
	fully_connected_layer(
		matrix* given_input,
		const matrix& input_format,
		int number_of_neurons,
		e_activation_t activation_function
	);

	/// <param name="given_input">the pointer where the input will be read</param>
	/// <param name="input_format">
	/// the first layer has usually a nullptr as input
	/// in order to set the right weights and biases, the input format is needed
	/// a format is an empty matrix with the right dimensions
	/// </param>
	/// <param name="activation_format">the format, that the output will have</param>
	/// <param name="activation_function">the activation function that will be used</param>
	fully_connected_layer(
		matrix* given_input,
		const matrix& input_format,
		const matrix& activation_format,
		e_activation_t activation_function
	);

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

	void forward_propagation() override;
	void back_propagation() override;

	void apply_deltas(int number_of_inputs) override;
};