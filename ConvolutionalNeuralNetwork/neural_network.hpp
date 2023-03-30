#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"

class neural_network {
private:
	matrix input;
	matrix* output;

	std::vector<std::unique_ptr<layer>> layer_types;
public:
	neural_network();

	void set_input(matrix input);
	matrix* get_output();

	void add_layer(const std::unique_ptr<layer> layer);

	void forward_propagation();
	void back_propagation(matrix* expected_output);
};