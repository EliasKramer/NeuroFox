#include "neural_network.hpp"

convolutional_layer* create_convolutional_layer(int kernel_size, int number_of_kernels, int stride, activation_function activation)
{
	return nullptr;
}

pooling_layer* create_pooling_layer(int size, int stride, pooling_type type)
{
	return nullptr;
}

fully_connected_layer* create_fully_connected_layer(int number_of_neurons, activation_function activation)
{
	return nullptr;
}

neural_network* create_neural_network(int input_size, int output_size, std::vector<convolutional_layer*> conv_layers, std::vector<pooling_layer*> pool_layers, std::vector<fully_connected_layer*> fc_layers)
{
	return nullptr;
}
