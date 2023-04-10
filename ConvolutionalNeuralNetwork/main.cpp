#include <iostream>
#include <vector>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"
int main()
{
	std::cout << "Hello World!" << std::endl;

	std::vector<std::unique_ptr<nn_data>> testing_data = 
		digit_data::get_digit_testing_data("..\\data\\digit_recognition");
	const std::vector<std::unique_ptr<nn_data>> training_data = 
		digit_data::get_digit_training_data("..\\data\\digit_recognition");
	
	//small traing data for testing

	neural_network nn;
	
	nn.set_input_format(get_matrix(28, 28, 1));
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.set_output_format(get_matrix(1, 10, 1));
	nn.add_last_fully_connected_layer(sigmoid_fn);
	nn.set_interpreter<digit_interpreter>();

	const digit_interpreter* interpreter = nn.get_interpreter<digit_interpreter>();
	std::cout << 
		"\nlabel: \n" <<
		interpreter->get_string_interpretation(training_data[0].get()->get_label_p())
		<< std::endl;

	nn.set_all_parameter(0);
	nn.apply_noise(0.1);

	nn.forward_propagation(training_data[0].get()->get_data_p());
	std::cout <<
		nn.get_interpreter<digit_interpreter>()->get_string_interpretation()
		<< std::endl;

	nn.learn(testing_data);

	nn.forward_propagation(training_data[0].get()->get_data_p());
	std::cout <<
		nn.get_interpreter<digit_interpreter>()->get_string_interpretation()
		<< std::endl;

	return 0;
}