#pragma once
#include "matrix.hpp"

enum _activation_function
{
	sigmoid,
	relu
} typedef activation_function;

enum _pooling_type
{
	max,
	min,
	average
} typedef pooling_type;

struct _fully_connected_layer {
	matrix* input;
	matrix weights;
	matrix biases;
	activation_function activation;
	matrix output;
} typedef fully_connected_layer;

struct _neural_kernel {
	matrix weights;
	matrix biases;
	matrix output;
} typedef neural_kernel;

struct _convolutional_layer {
	matrix* input;
	std::vector<neural_kernel> kernels;
	int stride;
	activation_function activation;
	matrix output;
} typedef convolutional_layer;

struct _pooling_layer {
	matrix* input;
	int kernel_width;
	int kernel_height;
	int stride;
	pooling_type type;
	matrix output;
} typedef pooling_layer;

struct _neural_network {
	matrix input;
	matrix output;

	//layer order is like this:
	//the first layer is always a convolutional layer
	//every convolutional layer has a pooling layer after it
	//they are always the same size
	//the last layers are always fully connected layers

	std::vector<convolutional_layer> convolutional_layers;
	std::vector<pooling_layer> pooling_layers;
	std::vector<fully_connected_layer> fully_connected_layers;
} typedef neural_network;

convolutional_layer* create_convolutional_layer(int kernel_size, int number_of_kernels, int stride, activation_function activation);
pooling_layer* create_pooling_layer(int size, int stride, pooling_type type);
fully_connected_layer* create_fully_connected_layer(int number_of_neurons, activation_function activation);

neural_network* create_neural_network(
	int input_size,
	int output_size,
	std::vector<convolutional_layer*> conv_layers,
	std::vector<pooling_layer*> pool_layers,
	std::vector<fully_connected_layer*> fc_layers
);