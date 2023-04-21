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
			matrix input(1, 1, 1);
			fully_connected_layer fc(1, e_activation_t::relu_fn);
			convolutional_layer c(2, 2, 1, e_activation_t::sigmoid_fn);
			pooling_layer p(2, 2, e_pooling_type_t::max_pooling);

			Assert::AreEqual((int)fc.get_layer_type(), (int)e_layer_type_t::fully_connected);
			Assert::AreEqual((int)c.get_layer_type(), (int)e_layer_type_t::convolution);
			Assert::AreEqual((int)p.get_layer_type(), (int)e_layer_type_t::pooling);
		}
	};
}