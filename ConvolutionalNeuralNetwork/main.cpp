#include <iostream>
#include "neural_network.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	matrix* input = create_matrix(1, 3, 1);
	matrix* output = create_matrix(1, 2, 1);

	neural_network nn;
	nn.set_input_format(*input);
	nn.set_input(input);
	nn.add_layer(
		std::make_unique<fully_connected_layer>
		(5, input, sigmoid_fn));
	nn.set_output_format(*output);
	nn.forward_propagation();
	return 0;
}