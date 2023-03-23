#pragma once

struct _fully_connected_layer {
	int input_size;
	int output_size;
	float* weights;
	float* biases;
} typedef fully_connected_layer;

struct _convolutional_layer {
	int input_width;
	int input_height;
	int input_depth;
	int filter_width;
	int filter_height;
	int filter_depth;
	int output_width;
	int output_height;
	int output_depth;
	float*weights;
	float*biases;
} typedef convolutional_layer;

struct _pooling_layer {
	int input_width;
	int input_height;
	int input_depth;
	int filter_width;
	int filter_height;
	int output_width;
	int output_height;
	int output_depth;
} typedef pooling_layer;

struct _neural_network {
	int input_width;
	int input_height;
	int input_depth;
	int output_size;
	int fully_connected_layer_count;
	int convolutional_layer_count;
	int pooling_layer_count;
	fully_connected_layer*fully_connected_layers;
	convolutional_layer*convolutional_layers;
	pooling_layer*pooling_layers;
} typedef neural_network;