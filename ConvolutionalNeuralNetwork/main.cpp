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

	//small traing data for testing

	neural_network nn;

	nn.set_input_format(get_matrix(28, 28, 1));
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.add_fully_connected_layer(16, sigmoid_fn);
	nn.set_output_format(get_matrix(1, 10, 1));
	nn.add_last_fully_connected_layer(sigmoid_fn);
	nn.set_interpreter<digit_interpreter>();

	nn.set_all_parameter(0);
	nn.apply_noise(0.1);

	std::cout << "start testing...\n";
	test_result result_before = nn.test(testing_data);
	std::cout << "result before training: \n" << result_before.to_string() << std::endl;

	while (true)
	{
		std::cout << "start training...\n";
		nn.learn(training_data, 60, 100);
		std::cout << "training done \n\n";

		std::cout << "start testing...\n";
		test_result result_after = nn.test(testing_data);
		std::cout << "result after training: \n" << result_after.to_string() << std::endl;
	}

	return 0;
}