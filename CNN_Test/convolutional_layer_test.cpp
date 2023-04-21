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
				layer.get_kernels_readonly()[0]
					.get_weights_readonly()
					.get_depth());
		}

	};
}