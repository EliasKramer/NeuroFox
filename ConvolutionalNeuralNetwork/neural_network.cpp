#include "neural_network.hpp"

neural_network* create_neural_network()
{
	neural_network* nn = new neural_network();
	resize_matrix(nn->input, 1, 1, 1);
	resize_matrix(nn->output, 1, 1, 1);
	return nn;
}
static bool can_add_layer_type(neural_network& nn, layer_type type)
{
	//no non-fully-connected layer can be added after a fully connected layer
	return
		(nn.layer_types.size() != 0 &&
			nn.layer_types.back() == fully_connected &&
			(type == pooling || type == convolution)) == false;
}
static matrix* get_last_output(neural_network& nn)
{
	layer_type last_layer_type = nn.layer_types.back();
	switch (last_layer_type)
	{
	case convolution:
		return &nn.convolutional_layers.back().output;
	case pooling:
		return &nn.pooling_layers.back().output;
	case fully_connected:
		return &nn.fully_connected_layers.back().output;
	default:
		throw "Unknown layer type";
	}
	return nullptr;
}

void add_fully_connected_layer(
	neural_network& nn,
	int number_of_neurons,
	activation activation
)
{
	//TODO

	//set the input for the layer layers
	//if there is no fully connected layer
	matrix* layer_input = get_last_output(nn);

	//create the layer
	fully_connected_layer layer_to_add =
		create_fully_connected_layer(
			number_of_neurons,
			layer_input,
			activation);

	//set the error of the previous layer
	//if there is a fully connected layer before
	if (nn.fully_connected_layers.size() != 0)
	{
		//set the right error of the last fully connected layer
		nn.fully_connected_layers.back().error_right =
			&layer_to_add.error;
	}

	//add the layer to the network
	nn.fully_connected_layers.push_back(layer_to_add);
	nn.layer_types.push_back(fully_connected);
}

void add_convolutional_layer(
	neural_network& nn,
	int kernel_size,
	int number_of_kernels,
	int stride)
{
	//set the input for the layer
	matrix* layer_input = nn.convolutional_layers.size() == 0
		//the network input is the input of layer
		? layer_input = &nn.input
		//the output of the last convolutional layer is the input of the layer
		: layer_input = &nn.convolutional_layers.back().output;

	//create the layer
	convolutional_layer layer_to_add =
		create_convolutional_layer(
			layer_input,
			kernel_size,
			number_of_kernels,
			stride,
			activation::relu);

	//set the error of the previous layer
	//if there is a convolutional layer before
	if (nn.convolutional_layers.size() != 0)
	{
		//set the right error of the last convolutional layer
		nn.convolutional_layers.back().error_right =
			&layer_to_add.error;
	}

	//add the layer to the network
	nn.convolutional_layers.push_back(layer_to_add);

	//add a pooling layer
	add_pooling_layer(
		nn,
		kernel_size,
		stride);
}

void add_pooling_layer(
	neural_network& nn,
	int kernel_size,
	int stride)
{
}

void set_input_format(neural_network& nn, matrix input_format)
{
	nn.input = input_format;
}

void set_output_format(neural_network& nn, matrix output_format)
{
	nn.output = output_format;
}

neural_network::neural_network()
{
}

void neural_network::set_input(matrix input)
{
}

matrix* neural_network::get_output()
{
	return nullptr;
}

void neural_network::add_layer(std::unique_ptr<layer> layer)
{
}

void neural_network::forward_propagation()
{
}

void neural_network::back_propagation(matrix* expected_output)
{
}

void neural_network::add_layer(layer_type layer_type)
{
}

void neural_network::add_layer(layer_type layer_type, int kernel_size, int number_of_kernels, int stride)
{
}

void neural_network::add_layer(layer_type layer_type, int number_of_neurons, activation activation)
{
}
