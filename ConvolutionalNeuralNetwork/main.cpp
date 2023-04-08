#include <iostream>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;
	
	std::vector<digit_data> training_data = digit_data::get_digit_testing_data("..\\data\\digit_recognition");

	matrix* input = create_matrix(28, 28, 1);
	set_all(*input, 1);
	matrix* output = create_matrix(1, 10, 1);

	neural_network nn;
	nn.set_input_format(*input);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.set_output_format(*output);
	nn.add_last_fully_connected_layer(sigmoid_fn);

	nn.set_interpreter(
		std::make_unique<digit_interpreter>(nn.get_output())
	);

	nn.forward_propagation(input);
	std::cout << nn.get_interpreter()->get_string_interpretation() << std::endl;
	
	nn.back_propagation(&training_data[0]);

	nn.forward_propagation(input);
	std::cout << nn.get_interpreter()->get_string_interpretation() << std::endl;
	
	return 0;
}