#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"

struct _fully_connected_layer {
	matrix* input;
	matrix output;

	matrix* previous_error;
	matrix error;

	matrix weights;
	matrix biases;
	
	matrix weight_deltas;
	matrix bias_deltas;

	activation activation_fn;
} typedef fully_connected_layer;

fully_connected_layer* create_fully_connected_layer(
	int number_of_neurons,
	matrix* input,
	activation activation);

void feed_forward(fully_connected_layer& layer);
void learn();