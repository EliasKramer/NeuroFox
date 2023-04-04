#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"

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
public:
	neural_network();

	//sets the input matrix to a certain format
	void set_input_format(const matrix& given_input_format);
	//sets the output matrix to a certain format
	void set_output_format(const matrix& given_output_format);
	
	void set_input(matrix* input);

	const matrix& get_output() const;

	void add_layer(std::unique_ptr<layer>&& layer);

	void forward_propagation();
	void back_propagation(const matrix& expected_output);
};