#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"

struct _neural_network {
	matrix input;
	matrix output;

	std::vector<layer_type> layer_types;

	std::vector<convolutional_layer> convolutional_layers;
	std::vector<pooling_layer> pooling_layers;
	std::vector<fully_connected_layer> fully_connected_layers;
} typedef neural_network;

neural_network* create_neural_network();

void add_fully_connected_layer(
	neural_network& nn,
	int number_of_neurons,
	activation activation
);
void add_convolutional_layer(
	neural_network& nn,
	int kernel_size,
	int number_of_kernels,
	int stride);
void add_pooling_layer(
	neural_network& nn,
	int kernel_size,
	int stride);

void set_input_format(
	neural_network& nn,
	matrix input_format
);

void set_output_format(
	neural_network& nn,
	matrix output_format);