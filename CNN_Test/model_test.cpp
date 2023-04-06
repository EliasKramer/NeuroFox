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
			std::vector<digit_data> training_data = digit_data::get_digit_training_data("..\\..\\data\\digit_recognition");
			std::vector<digit_data> testing_data = digit_data::get_digit_testing_data("..\\..\\data\\digit_recognition");

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

			std::string expected_output = 
				std::string("1: 0.500000\n") +
				std::string("2: 0.500000\n") +
				std::string("3: 0.500000\n") +
				std::string("4: 0.500000\n") +
				std::string("5: 0.500000\n") +
				std::string("6: 0.500000\n") +
				std::string("7: 0.500000\n") +
				std::string("8: 0.500000\n") +
				std::string("9: 0.500000\n") +
				std::string("10: 0.500000\n") +
				std::string("result: 1\n");

			Assert::AreEqual(expected_output, nn.get_interpreter()->get_string_interpretation());
		}
	};
}
