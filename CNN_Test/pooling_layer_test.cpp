#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/pooling_layer.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(pooling_test)
	{
	public:

		TEST_METHOD(pooling_constructor_test_1)
		{
			pooling_layer pooling(1, 1, max_pooling);
			pooling.set_input_format(vector3(2, 2, 1));

			Assert::AreEqual((size_t)2, pooling.get_activations_readonly().get_height());
			Assert::AreEqual((size_t)2, pooling.get_activations_readonly().get_width());
			Assert::AreEqual((size_t)1, pooling.get_activations_readonly().get_depth());

			Assert::AreEqual((size_t)1, pooling.get_stride());
			Assert::AreEqual((size_t)1, pooling.get_filter_size());
			Assert::AreEqual((int)max_pooling, (int)pooling.get_pooling_fn());
		}
		TEST_METHOD(pooling_constructor_test_2)
		{
			pooling_layer pooling(2, 2, average_pooling);
			pooling.set_input_format(vector3(6,6,2));

			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_height());
			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_width());
			Assert::AreEqual((size_t)2, pooling.get_activations_readonly().get_depth());

			Assert::AreEqual((size_t)2, pooling.get_stride());
			Assert::AreEqual((size_t)2, pooling.get_filter_size());
			Assert::AreEqual((int)average_pooling, (int)pooling.get_pooling_fn());
		}
		TEST_METHOD(pooling_constructor_test_invalid_arguments)
		{
			matrix input(vector3(6, 6, 2));

			//wrong filter
			try
			{
				pooling_layer pooling(0, 2, average_pooling);
				Assert::Fail();
			}
			catch (std::invalid_argument e)
			{
				Assert::AreEqual("filter size must be greater than 0", e.what());
			}

			//wrong stride
			try
			{
				pooling_layer pooling(2, 0, average_pooling);
				Assert::Fail();
			}
			catch (std::invalid_argument e)
			{
				Assert::AreEqual("stride must be greater than 0", e.what());
			}
		}
		TEST_METHOD(feed_forward_test_min_pooling)
		{
			matrix input(vector3(6, 6, 2));
			input.set_all(1);
			input.set_at(vector3(0, 0, 0), 0);

			pooling_layer pooling(2, 2, min_pooling);
			pooling.set_input_format(input.get_format());

			pooling.forward_propagation(input);
			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_height());
			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_width());
			Assert::AreEqual((size_t)2, pooling.get_activations_readonly().get_depth());

			Assert::AreEqual(0.0f, pooling.get_activations_readonly().get_at_flat_host(0));
			for (int i = 1; i < pooling.get_activations_readonly().item_count(); i++)
			{
				Assert::AreEqual(1.0f, pooling.get_activations_readonly().get_at_flat_host(i));
			}
		}
		TEST_METHOD(feed_forward_test_min_pooling_gpu)
		{
			matrix input(vector3(6, 6, 2));
			input.enable_gpu_mode();
			input.set_all(1);
			input.set_at(vector3(0, 0, 0), 0);
			input.sync_device_and_host();

			pooling_layer pooling(2, 2, min_pooling);
			pooling.set_input_format(input.get_format());
			pooling.enable_gpu_mode();

			pooling.forward_propagation(input);
			pooling.sync_device_and_host();
			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_height());
			Assert::AreEqual((size_t)3, pooling.get_activations_readonly().get_width());
			Assert::AreEqual((size_t)2, pooling.get_activations_readonly().get_depth());

			Assert::AreEqual(0.0f, pooling.get_activations_readonly().get_at_flat_host(0));
			for (int i = 1; i < pooling.get_activations_readonly().item_count(); i++)
			{
				Assert::AreEqual(1.0f, pooling.get_activations_readonly().get_at_flat_host(i));
			}
		}
	};
}