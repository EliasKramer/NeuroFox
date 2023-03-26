#include <iostream>
#include "neural_network.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	convolutional_layer* conv_layer = create_convolutional_layer(3, 32, 1, relu);
	pooling_layer* pool_layer = create_pooling_layer(2, 2, max);
	fully_connected_layer* fc_layer = create_fully_connected_layer(10, sigmoid);

	std::vector<convolutional_layer*> conv_layers;
	conv_layers.push_back(conv_layer);
	std::vector<pooling_layer*> pool_layers;
	pool_layers.push_back(pool_layer);
	std::vector<fully_connected_layer*> fc_layers;
	fc_layers.push_back(fc_layer);

	neural_network* nn = create_neural_network(28 * 28, 10, conv_layers, pool_layers, fc_layers);
	
	return 0;
}