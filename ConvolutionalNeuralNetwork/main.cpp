#include <iostream>
#include <vector>
#include "neural_network.hpp"
#include "digit_interpreter.hpp"
#include "digit_data.hpp"
int main()
{
	/*
	std::cout << "Hello World!" << std::endl;
	std::cout << "Loading data..." << std::endl << std::endl;
	std::vector<std::unique_ptr<nn_data>> testing_data =
		digit_data::get_digit_testing_data("..\\data\\digit_recognition");
	const std::vector<std::unique_ptr<nn_data>> training_data =
		digit_data::get_digit_training_data("..\\data\\digit_recognition");

	std::cout << std::endl << "data loaded" << std::endl << std::endl;

	neural_network nn;

	nn.set_input_format(matrix(28, 28, 1));
	//nn.add_convolutional_layer(1, 7, 1, sigmoid_fn);
	nn.add_fully_connected_layer(25, sigmoid_fn);
	nn.add_fully_connected_layer(25, sigmoid_fn);
	nn.set_output_format(matrix(1, 10, 1));
	nn.add_last_fully_connected_layer(sigmoid_fn);
	nn.set_interpreter<digit_interpreter>();
	nn.enable_gpu();

	nn.set_all_parameter(0);
	nn.apply_noise(0.1f);

	nn.forward_propagation(testing_data[0].get()->get_data_p());

	std::cout << "guessed_label" << std::endl
		<< nn.get_interpreter<digit_interpreter>()->get_string_interpretation(nn.get_output())
		<< std::endl;
	*/
	std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7
	};
	matrix input(input_data, 4, 4, 1);

	std::vector<float> kernel_data = {
		1, 2,
		3, 4
	};
	matrix kernel(kernel_data, 2, 2, 1);

	matrix expected(2, 2, 1);
	expected.set_at(0, 0,
		1 * 1 + 2 * 2 +
		3 * 5 + 4 * 6);
	expected.set_at(1, 0,
		1 * 3 + 2 * 4 +
		3 * 7 + 4 * 8);
	expected.set_at(0, 1,
		1 * 9 + 2 * 1 +
		3 * 4 + 4 * 5);
	expected.set_at(1, 1,
		1 * 2 + 2 * 3 +
		3 * 6 + 4 * 7);

	matrix double_check_expected(2, 2, 1);
	matrix::valid_cross_correlation(input, kernel, double_check_expected, 2);

	gpu_memory<float> gpu_input(input);
	gpu_memory<float> gpu_kernel(kernel);
	gpu_memory<float> gpu_result(4);

	std::vector<gpu_memory<float>> gpu_kernel_weights = { gpu_kernel };

	gpu_valid_cross_correlation(
		gpu_input,
		gpu_kernel_weights,
		gpu_result,
		input.get_width(),
		input.get_depth(),
		kernel.get_width(),
		gpu_kernel_weights.size(),
		2, //stride 
		2); //output width

	std::vector<float> result = *gpu_result.to_cpu().get();
	matrix result_matrix = matrix(result, 2, 2, 1);

	std::cout << "expected" << std::endl << expected.get_string() << std::endl;

	return 0;
}