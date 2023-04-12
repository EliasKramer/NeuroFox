#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/neural_network.hpp"
#include "../ConvolutionalNeuralNetwork/digit_data.hpp"
#include "../ConvolutionalNeuralNetwork/digit_interpreter.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(model_test)
	{
	public:

		TEST_METHOD(testing_3blue_1brown_model)
		{
			std::vector<std::unique_ptr<nn_data>> testing_data =
				digit_data::get_digit_testing_data("..\\..\\data\\digit_recognition");

			neural_network nn;

			nn.set_input_format(get_matrix(28, 28, 1));
			nn.add_fully_connected_layer(25, sigmoid_fn);
			nn.add_fully_connected_layer(25, sigmoid_fn);
			nn.set_output_format(get_matrix(1, 10, 1));
			nn.add_last_fully_connected_layer(sigmoid_fn);
			nn.set_interpreter<digit_interpreter>();

			nn.set_all_parameter(0);

			nn.learn_once(testing_data[0], true);
			nn.forward_propagation(testing_data[0].get()->get_data_p());
			
			Assert::IsTrue(
				nn.get_interpreter<digit_interpreter>()->same_result(
					*nn.get_output(), 
					testing_data[0].get()->get_label()));
		}
	};
}
