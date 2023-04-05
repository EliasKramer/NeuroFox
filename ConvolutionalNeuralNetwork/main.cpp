#include <iostream>
#include "neural_network.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	matrix* input = create_matrix(28, 28, 1);
	matrix* output = create_matrix(1, 10, 1);

	neural_network nn;
	nn.set_input_format(*input);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.set_output_format(*output);

	nn.forward_propagation(input);
	return 0;
}