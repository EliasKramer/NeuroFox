#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/convolutional_layer.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(convolutional_layer_test)
	{
	public:

		TEST_METHOD(constructor_test)
		{
			convolutional_layer layer(3, 2, 1, e_activation_t::relu_fn);
			Assert::AreEqual((int)e_layer_type_t::convolution, (int)layer.get_layer_type());
			Assert::AreEqual(3, (int)layer.get_kernel_count());
			Assert::AreEqual(2, layer.get_kernel_size());
			Assert::AreEqual(1, layer.get_stride());
		}
		TEST_METHOD(invalid_constructor)
		{
			try
			{
				convolutional_layer layer(0, 2, 1, e_activation_t::relu_fn);
				Assert::Fail();
			}
			catch (std::invalid_argument& e)
			{
				Assert::AreEqual("number_of_kernels must be greater than 0", e.what());
			}
			try
			{
				convolutional_layer layer(3, 0, 1, e_activation_t::relu_fn);
				Assert::Fail();
			}
			catch (std::invalid_argument& e)
			{
				Assert::AreEqual("kernel_size must be greater than 0", e.what());
			}
			try
			{
				convolutional_layer layer(3, 2, 0, e_activation_t::relu_fn);
				Assert::Fail();
			}
			catch (std::invalid_argument& e)
			{
				Assert::AreEqual("stride must be greater than 0", e.what());
			}
			try
			{
				convolutional_layer layer(3, 2, 3, e_activation_t::relu_fn);
				Assert::Fail();
			}
			catch (std::invalid_argument& e)
			{
				Assert::AreEqual("stride must be smaller or equal than the kernel_size", e.what());
			}
		}
		TEST_METHOD(set_input_format_test)
		{
			convolutional_layer layer(3, 2, 1, e_activation_t::relu_fn);
			matrix input_format(5, 5, 6);
			layer.set_input_format(input_format);
			//the width and the heigh are
			//the (input_size - kernel_size) / stride + 1
			Assert::AreEqual(4, layer.get_activations().get_width());
			Assert::AreEqual(4, layer.get_activations().get_height());
			//the depth is the number of kernels
			Assert::AreEqual(3, layer.get_activations().get_depth());

			//the kernel depth is the input depth
			Assert::AreEqual(
				6, 
				layer
					.get_kernel_weights_readonly()[0]
					.get_depth());
		}
		TEST_METHOD(feed_forward_test)
		{
			convolutional_layer layer(1, 2, 1, e_activation_t::sigmoid_fn);
			matrix input(3, 3, 1);
			layer.set_input_format(input);

			/* weight matrix
				+ - + - +
				| 1 | 3 |
				+ - + - +
				| 2 | 4 |
				+ - + - +
			*/
			layer.get_kernel_weights()[0].set_at(0, 1, 2.0f);
			layer.get_kernel_weights()[0].set_at(1, 0, 3.0f);
			layer.get_kernel_weights()[0].set_at(1, 1, 4.0f);
			layer.get_kernel_weights()[0].set_at(0, 0, 1.0f);

			/* bias matrix
				+ - + - +
				|-60|-60|
				+ - + - +
				|-60|-60|
				+ - + - +
			*/
			layer.get_kernel_biases().set_all(-60);

			/* input matrix
				+ - + - + - +
				| 1 | 4 | 7 |
				+ - + - + - +
				| 2 | 5 | 8 |
				+ - + - + - +
				| 3 | 6 | 9 |
				+ - + - + - +
			*/
			input.set_at(0, 0, 1);
			input.set_at(0, 1, 2);
			input.set_at(0, 2, 3);
			input.set_at(1, 0, 4);
			input.set_at(1, 1, 5);
			input.set_at(1, 2, 6);
			input.set_at(2, 0, 7);
			input.set_at(2, 1, 8);
			input.set_at(2, 2, 9);

			layer.forward_propagation_cpu(input);

			/* expected output matrix
				+ -- + -- +
				|-23 | 07 |
				+ -- + -- +
				|-13 | 17 |
				+ -- + -- +
			*/

			//but the matrix did an activation function. in this case sigmoid
			//so the output matrix is
			/*
					(+ -- + -- +)
					(|-23 | 07 |)
			sigmoid (+ -- + -- +)
					(|-13 | 17 |)
					(+ -- + -- +)
			*/

			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](-23.0f),
				layer.get_activations().get_at(0, 0));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](7.0f),
				layer.get_activations().get_at(1, 0));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](-13.0f),
				layer.get_activations().get_at(0, 1));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](17.0f),
				layer.get_activations().get_at(1, 1));
		}
	};
}