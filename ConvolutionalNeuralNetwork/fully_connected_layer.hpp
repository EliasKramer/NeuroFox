#pragma once
#include "matrix.hpp"
#include "layer.hpp"

struct _fully_connected_layer {
	matrix* input;
	matrix weights;
	matrix biases;
	activation_function activation;
	matrix output;
} typedef fully_connected_layer;

fully_connected_layer* create_fully_connected_layer(int number_of_neurons, activation_function activation);