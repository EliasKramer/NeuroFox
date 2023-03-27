#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"


struct _neural_network {
	matrix input;
	matrix* output;

	//layer order is like this:
	//the first layer is always a convolutional layer
	//every convolutional layer has a pooling layer after it
	//they are always the same size
	//the last layers are always fully connected layers

	std::vector<convolutional_layer> convolutional_layers;
	std::vector<pooling_layer> pooling_layers;
	std::vector<fully_connected_layer> fully_connected_layers;
} typedef neural_network;

neural_network* create_neural_network(
	matrix input,
	matrix output,
	std::vector<convolutional_layer> conv_layers,
	std::vector<pooling_layer> pool_layers,
	std::vector<fully_connected_layer> fc_layers
);