#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "interpreter.hpp"
class neural_network {
private:
	matrix* input_p = nullptr;
	matrix* output_p = nullptr;

	//the input format is the width, height and depth of the input
	matrix input_format;
	//the input matrix can only be set once
	bool input_format_set = false;

	//the output format is the width, height and depth of the output
	matrix output_format;
	//the output matrix can only be set once
	bool output_format_set = false;

	std::vector<std::unique_ptr<layer>> layers;

	//the hash of the last input matrix that propagated forward
	size_t last_processed_input_hash = 0;

	std::unique_ptr<interpreter> interpreter_p = 0;

	void set_input(matrix* input);

	layer* get_last_layer();

	matrix* get_last_layer_output();
	matrix* get_last_layer_format();

	void add_layer(std::unique_ptr<layer>&& given_layer);

public:
	neural_network();

	//sets the input matrix to a certain format
	void set_input_format(const matrix& given_input_format);
	//sets the output matrix to a certain format
	void set_output_format(const matrix& given_output_format);

	const matrix& get_output() const;

	void add_fully_connected_layer(int num_neurons, e_activation_t activation_fn);
	void add_convolutional_layer(int kernel_size, int number_of_kernels, int stride, e_activation_t activation_fn);
	void add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type);
	
	void add_last_fully_connected_layer(e_activation_t activation_fn);

	void set_interpreter(std::unique_ptr<interpreter>&& given_interpreter);
	const interpreter* get_interpreter() const;

	void forward_propagation(matrix* input);
	void back_propagation(const matrix& expected_output);
};