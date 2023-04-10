#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "interpreter.hpp"
#include "nn_data.hpp"

class neural_network {
private:
	matrix* output_p = nullptr;
	matrix cost_derivative;

	//the input format is the width, height and depth of the input
	matrix input_format;
	//the input matrix can only be set once
	bool input_format_set = false;

	//the output format is the width, height and depth of the output
	matrix output_format;
	//the output matrix can only be set once
	bool output_format_set = false;

	std::vector<std::unique_ptr<layer>> layers;
	//saves the indices of all layers tha have parameter
	//convolutional and fully connected 
	std::vector<int> parameter_layer_indices;

	//the hash of the last input matrix that propagated forward
	size_t last_processed_input_hash = 0;

	std::unique_ptr<interpreter> interpreter_p = 0;

	void set_input(const matrix* input);

	layer* get_last_layer();

	void add_layer(std::unique_ptr<layer>&& given_layer);

	//we need the training_data_count for 
	//calculating the average of the deltas
	void apply_deltas(int training_data_count);

public:
	neural_network();

	//sets the input matrix to a certain format
	void set_input_format(const matrix& given_input_format);
	//sets the output matrix to a certain format
	void set_output_format(const matrix& given_output_format);

	const matrix* get_output() const;

	void add_fully_connected_layer(int num_neurons, e_activation_t activation_fn);
	void add_convolutional_layer(int kernel_size, int number_of_kernels, int stride, e_activation_t activation_fn);
	void add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type);

	void add_last_fully_connected_layer(e_activation_t activation_fn);

	template<typename interpreter_type>
	void set_interpreter()
	{
		if (std::is_base_of<interpreter, interpreter_type>::value == false)
		{
			throw std::runtime_error(
				"Cannot set interpreter. Interpreter type is not a child of interpreter.");
		}
		interpreter_p = std::make_unique<interpreter_type>(get_output());
	}

	template<typename interpreter_type>
	const interpreter_type* get_interpreter() const
	{
		if (interpreter_p == nullptr)
		{
			throw std::runtime_error("Cannot get interpreter. Interpreter is not set yet.");
		}

		if (std::is_base_of<interpreter, interpreter_type>::value == false)
		{
			throw std::runtime_error(
				"Cannot get interpreter. Interpreter type is not a child of interpreter.");
		}

		return dynamic_cast<interpreter_type*>(interpreter_p.get());
	}

	//set all weights and biases to that value
	void set_all_parameter(float value);
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range);
	//add a random value between range and -range to one weight or bias 
	void mutate(float range);

	float test(std::vector<nn_data>& test_data);
	void forward_propagation(const matrix* input);

	void learn(const std::vector<std::unique_ptr<nn_data>>& training_data);
	void learn_once(const std::unique_ptr<nn_data>& expected_output, bool apply_changes = true);
};