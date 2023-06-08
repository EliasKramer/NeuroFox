#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/neural_network.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(neural_network_test)
	{
	public:

		TEST_METHOD(nn_equal_format_test)
		{
			neural_network nn;
			nn.set_input_format(vector3(28, 28, 1));
			nn.add_fully_connected_layer(15, e_activation_t::relu_fn);
			nn.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			nn.set_all_parameters(1.0f);

			neural_network same;
			same.set_input_format(vector3(28, 28, 1));
			same.add_fully_connected_layer(15, e_activation_t::relu_fn);
			same.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			same.set_all_parameters(1.0f);
			same.mutate(5);

			neural_network not_same_1;
			not_same_1.set_input_format(vector3(28, 28, 1));
			not_same_1.add_fully_connected_layer(15, e_activation_t::relu_fn);
			not_same_1.add_fully_connected_layer(10, e_activation_t::relu_fn);
			not_same_1.set_all_parameters(1.0f);

			neural_network not_same_2;
			not_same_2.set_input_format(vector3(28, 28, 1));
			not_same_2.add_fully_connected_layer(14, e_activation_t::relu_fn);
			not_same_2.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			not_same_2.set_all_parameters(1.0f);

			neural_network not_same_3;
			not_same_3.set_input_format(vector3(28, 28, 1));
			not_same_3.add_fully_connected_layer(15, e_activation_t::relu_fn);
			not_same_3.set_all_parameters(1.0f);

			Assert::AreEqual(true, nn.equal_format(same));
			Assert::AreEqual(false, nn.equal_format(not_same_1));
			Assert::AreEqual(false, nn.equal_format(not_same_2));
			Assert::AreEqual(false, nn.equal_format(not_same_3));
		}
		TEST_METHOD(nn_equal_parameter_test)
		{
			neural_network nn;
			nn.set_input_format(vector3(28, 28, 1));
			nn.add_fully_connected_layer(15, e_activation_t::relu_fn);
			nn.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			nn.set_all_parameters(1.0f);

			neural_network same;
			same.set_input_format(vector3(28, 28, 1));
			same.add_fully_connected_layer(15, e_activation_t::relu_fn);
			same.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			same.set_all_parameters(1.0f);

			neural_network not_same_1;
			not_same_1.set_input_format(vector3(28, 28, 1));
			not_same_1.add_fully_connected_layer(15, e_activation_t::relu_fn);
			not_same_1.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			not_same_1.set_all_parameters(1.0f);
			not_same_1.mutate(5);

			neural_network not_same_2;
			not_same_2.set_input_format(vector3(28, 28, 1));
			not_same_2.add_fully_connected_layer(14, e_activation_t::relu_fn);
			not_same_2.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			not_same_2.set_all_parameters(1.0f);

			Assert::AreEqual(true, nn.equal_parameter(same));
			Assert::AreEqual(false, nn.equal_parameter(not_same_1));
			Assert::AreEqual(false, nn.equal_parameter(not_same_2));
		}
		TEST_METHOD(nn_copy_constructor_test)
		{
			neural_network nn;
			nn.set_input_format(vector3(28, 28, 1));
			nn.add_fully_connected_layer(15, e_activation_t::relu_fn);
			nn.add_fully_connected_layer(10, e_activation_t::sigmoid_fn);
			nn.set_all_parameters(1.0f);

			neural_network copy(nn);
			Assert::AreEqual(true, nn.equal_format(copy));
			Assert::AreEqual(true, nn.equal_parameter(copy));

			copy.mutate(5);
			Assert::AreEqual(false, nn.equal_parameter(copy));
			Assert::AreEqual(true, nn.equal_format(copy));
		}
	};
}
