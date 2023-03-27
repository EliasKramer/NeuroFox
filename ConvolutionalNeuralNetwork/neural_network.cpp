#include "neural_network.hpp"

neural_network* create_neural_network(
	matrix input, 
	matrix output, 
	std::vector<convolutional_layer> conv_layers, 
	std::vector<pooling_layer> pool_layers, 
	std::vector<fully_connected_layer> fc_layers)
{
	//for every convolutional layer, there should be a pooling layer
	if(conv_layers.size() != pool_layers.size())
	{
		throw "convolutional layers and pooling layers are not the same size";
	}
	if (fc_layers.size() == 0)
	{
		throw "there should be at least one fully connected layer";
	}

	neural_network* nn = new neural_network;

	//copy the input
	nn->input = input;

	//copy every layer
	std::copy(conv_layers.begin(), conv_layers.end(), 
		nn->convolutional_layers.begin());
	std::copy(pool_layers.begin(), pool_layers.end(), 
		nn->pooling_layers.begin());
	std::copy(fc_layers.begin(), fc_layers.end(), 
		nn->fully_connected_layers.begin());

	//setting the input of the first layer to the input of the neural network
	if (conv_layers.size() == 0)
	{
		//if there are no convolutional layers, 
		//then the first layer is a fully connected layer
		fc_layers[0].input = &nn->input;
	}
	else 
	{
		//if there are convolutional layers,
		//then the first layer is a convolutional layer
		nn->convolutional_layers[0].input = &nn->input;
		nn->pooling_layers[0].input = &nn->convolutional_layers[0].output;
	}

	//TODO

	return nn;
}
