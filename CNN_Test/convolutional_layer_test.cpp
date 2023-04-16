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
			Assert::AreEqual(3, (int)layer.get_kernels_readonly().size());
			Assert::AreEqual(2, layer.get_kernel_size());
			Assert::AreEqual(1, layer.get_stride());
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
				layer.get_kernels_readonly()[0]
					.get_weights_readonly()
					.get_depth());
		}
		TEST_METHOD(single_kernel_test)
		{
			conv_kernel kernel(2);
			kernel.set_kernel_depth(1);

			/* weight matrix
				+ - + - +
				| 1 | 3 |
				+ - + - +
				| 2 | 4 |
				+ - + - +
			*/

			kernel.get_weights().set_at(0, 0, 1);
			kernel.get_weights().set_at(0, 1, 2);
			kernel.get_weights().set_at(1, 0, 3);
			kernel.get_weights().set_at(1, 1, 4);

			kernel.set_bias(5);

			matrix input(3, 3, 1);

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

			/*
				+ - + - + - +	+ - + - + - +
				| 1 | 4 | 7 |	| 1 | 3 |	|
				+ - + - + - +	+ - + - + - +
				| 2 | 5 | 8 | * | 2 | 4 |	| + 5 = 1*1 + 4*3 + 2*2 + 5*4 + 5 = 42
				+ - + - + - +	+ - + - + - +
				| 3 | 6 | 9 |	|	|	|	|
				+ - + - + - +	+ - + - + - +
			*/
			float result = kernel.lay_kernel_over_matrix(input, 0, 0);
			Assert::AreEqual(42.0f, result);

			/*
				+ - + - + - +	+ - + - + - +
				| 1 | 4 | 7 |	|	| 1 | 3 |
				+ - + - + - +	+ - + - + - +
				| 2 | 5 | 8 | * |	| 2 | 4 | + 5 = 1*4 + 3*7 + 2*5 + 4*8 + 5 = 72
				+ - + - + - +	+ - + - + - +
				| 3 | 6 | 9 |	|	|	|	|
				+ - + - + - +	+ - + - + - +
			*/
			result = kernel.lay_kernel_over_matrix(input, 1, 0);
			Assert::AreEqual(72.0f, result);

			/*
				+ - + - + - +	+ - + - + - +
				| 1 | 4 | 7 |	|	|	|	|
				+ - + - + - +	+ - + - + - +
				| 2 | 5 | 8 | * | 1 | 3 |	| + 5 = 1*2 + 3*5 + 2*3 + 4*6 + 5 = 52
				+ - + - + - +	+ - + - + - +
				| 3 | 6 | 9 |	| 2 | 4 |	|
				+ - + - + - +	+ - + - + - +
			*/
			result = kernel.lay_kernel_over_matrix(input, 0, 1);
			Assert::AreEqual(52.0f, result);


			/*
				+ - + - + - +	+ - + - + - +
				| 1 | 4 | 7 |	|	|	|	|
				+ - + - + - +	+ - + - + - +
				| 2 | 5 | 8 | * |	| 1 | 3 | + 5 = 1*5 + 3*8 + 2*6 + 4*9 + 5 = 82
				+ - + - + - +	+ - + - + - +
				| 3 | 6 | 9 |	|	| 2 | 4 |
				+ - + - + - +	+ - + - + - +
			*/
			result = kernel.lay_kernel_over_matrix(input, 1, 1);
			Assert::AreEqual(82.0f, result);
		}

		
		TEST_METHOD(feed_forward_test)
		{
			convolutional_layer layer(1, 2, 1, e_activation_t::sigmoid_fn);
			matrix input(3, 3, 1);
			layer.set_input_format(input);
			layer.set_input(&input);

			/* weight matrix
				+ - + - +
				| 1 | 3 |
				+ - + - +
				| 2 | 4 |
				+ - + - +
			*/
			layer.get_kernels()[0].get_weights().set_at(0, 0, 1.0f);
			layer.get_kernels()[0].get_weights().set_at(0, 1, 2.0f);
			layer.get_kernels()[0].get_weights().set_at(1, 0, 3.0f);
			layer.get_kernels()[0].get_weights().set_at(1, 1, 4.0f);

			layer.get_kernels()[0].set_bias(5);

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

			layer.forward_propagation();

			/* expected output matrix
				+ -- + -- +
				| 42 | 72 |
				+ -- + -- +
				| 52 | 82 |
				+ -- + -- +
			*/
			//but the matrix did an activation function. in this case sigmoid
			//so the output matrix is
			/*
					(+ -- + -- +)
					(| 42 | 72 |)
			sigmoid (+ -- + -- +)
					(| 52 | 82 |)
					(+ -- + -- +)
			*/
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](42.0f), 
				layer.get_activations().matrix_get_at(0, 0));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](72.0f),
				layer.get_activations().matrix_get_at(1, 0));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](52.0f),
				layer.get_activations().matrix_get_at(0, 1));
			Assert::AreEqual(
				ACTIVATION[sigmoid_fn](82.0f),
				layer.get_activations().matrix_get_at(1, 1));
		}
	};
}