#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/layer.hpp"
#include "../ConvolutionalNeuralNetwork/fully_connected_layer.hpp"
#include "../ConvolutionalNeuralNetwork/convolutional_layer.hpp"
#include "../ConvolutionalNeuralNetwork/pooling_layer.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(layer_test)
	{
	public:
		TEST_METHOD(layer_gets_right_type_in_constructor)
		{
			matrix input(vector3(1, 1, 1));
			fully_connected_layer fc = fully_connected_layer((size_t)1, e_activation_t::relu_fn);
			convolutional_layer c(2, 2, 1, e_activation_t::sigmoid_fn);
			pooling_layer p(2, 2, e_pooling_type_t::max_pooling);

			Assert::AreEqual((int)fc.get_layer_type(), (int)e_layer_type_t::fully_connected);
			Assert::AreEqual((int)c.get_layer_type(), (int)e_layer_type_t::convolution);
			Assert::AreEqual((int)p.get_layer_type(), (int)e_layer_type_t::pooling);
		}
		TEST_METHOD(layer_equal_format_input_test)
		{
			fully_connected_layer start((size_t)15, e_activation_t::relu_fn);
			start.set_input_format(vector3(3, 5, 2));

			fully_connected_layer same((size_t)15, e_activation_t::relu_fn);
			same.set_input_format(vector3(3, 5, 2));

			fully_connected_layer not_same((size_t)15, e_activation_t::relu_fn);
			not_same.set_input_format(vector3(3, 6, 2));

			Assert::AreEqual(start.equal_format(not_same), false);
			Assert::AreEqual(start.equal_format(same), true);
		}
		TEST_METHOD(layer_equal_format_activation_fn_test)
		{
			convolutional_layer start(3, 2, 2, e_activation_t::sigmoid_fn);
			start.set_input_format(vector3(4, 4, 1));

			convolutional_layer same(3, 2, 2, e_activation_t::sigmoid_fn);
			same.set_input_format(vector3(4, 4, 1));

			convolutional_layer not_same(3, 2, 2, e_activation_t::relu_fn);
			not_same.set_input_format(vector3(4, 4, 1));

			Assert::AreEqual(start.equal_format(not_same), false);
			Assert::AreEqual(start.equal_format(same), true);
		}
		TEST_METHOD(layer_equal_format_pooling_layer_test)
		{
			pooling_layer start(2, 2, e_pooling_type_t::max_pooling);
			start.set_input_format(vector3(4, 4, 1));
			pooling_layer same(2, 2, e_pooling_type_t::max_pooling);
			same.set_input_format(vector3(4, 4, 1));
			pooling_layer not_same(2, 2, e_pooling_type_t::average_pooling);
			not_same.set_input_format(vector3(4, 4, 1));

			Assert::AreEqual(start.equal_format(not_same), false);
			Assert::AreEqual(start.equal_format(same), true);
		}
		TEST_METHOD(layer_equal_parameter_same_type_test)
		{
			fully_connected_layer start(1, e_activation_t::relu_fn);
			start.set_input_format(vector3(1, 1, 1));
			fully_connected_layer same(1, e_activation_t::relu_fn);
			same.set_input_format(vector3(1, 1, 1));
			convolutional_layer not_same(1, 1, 1, e_activation_t::relu_fn);
			not_same.set_input_format(vector3(1, 1, 1));
			
			Assert::AreEqual(start.equal_format(not_same), false);
			Assert::AreEqual(start.equal_format(same), true);
		}
		TEST_METHOD(layer_equal_parameter_mutated_test)
		{
			fully_connected_layer start(5, e_activation_t::relu_fn);
			start.set_input_format(vector3(1, 1, 1));
			start.set_all_parameters(1.0f);
			fully_connected_layer same(5, e_activation_t::relu_fn);
			same.set_input_format(vector3(1, 1, 1));
			same.set_all_parameters(1.0f);
			fully_connected_layer not_same(5, e_activation_t::sigmoid_fn);
			not_same.set_input_format(vector3(1, 1, 1));
			not_same.set_all_parameters(1.0f);
			not_same.mutate(1.0f);
			
			Assert::AreEqual(start.equal_parameter(not_same), false);
			Assert::AreEqual(start.equal_parameter(same), true);
		}
	};
}