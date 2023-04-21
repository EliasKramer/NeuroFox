#include <iostream>
#include <vector>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;
	std::cout << "Loading data..." << std::endl << std::endl;
	std::vector<std::unique_ptr<nn_data>> testing_data =
		digit_data::get_digit_testing_data("..\\data\\digit_recognition");
	const std::vector<std::unique_ptr<nn_data>> training_data =
		digit_data::get_digit_training_data("..\\data\\digit_recognition");

	std::cout << std::endl << "data loaded" << std::endl << std::endl;

	neural_network nn;

	nn.set_input_format(matrix(28, 28, 1));
	nn.add_convolutional_layer(1, 7, 1, sigmoid_fn);
	nn.add_fully_connected_layer(25, sigmoid_fn);
	nn.add_fully_connected_layer(25, sigmoid_fn);
	nn.set_output_format(matrix(1, 10, 1));
	nn.add_last_fully_connected_layer(sigmoid_fn);
	nn.set_interpreter<digit_interpreter>();

	nn.set_all_parameter(0);
	nn.apply_noise(0.1f);

	nn.forward_propagation(testing_data[0].get()->get_data_p());

	std::cout << "guessed_label" << std::endl
		<< nn.get_interpreter<digit_interpreter>()->get_string_interpretation(nn.get_output())
		<< std::endl;
	
	return 0;
}