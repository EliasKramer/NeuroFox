#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"

struct _fully_connected_layer {
	matrix* input;
	matrix output;

	matrix weights;
	matrix biases;
	activation activation_fn;
} typedef fully_connected_layer;

fully_connected_layer* create_fully_connected_layer(
	int number_of_neurons,
	activation activation,
	matrix* input
);

void forward_propagate_fully_connected_layer(
	fully_connected_layer& layer);