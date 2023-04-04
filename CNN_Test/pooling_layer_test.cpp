#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/pooling_layer.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(pooling_test)
	{
	public:

		TEST_METHOD(pooling_constructor_test_1)
		{
			matrix* input = create_matrix(2, 2, 1);
			pooling_layer pooling(input , 1, 1, max_pooling);
			Assert::AreEqual(2, pooling.get_activations().height);
			Assert::AreEqual(2, pooling.get_activations().width);
			Assert::AreEqual(1, pooling.get_activations().depth);

			Assert::IsTrue(input == pooling.get_input_p());
			Assert::AreEqual(1, pooling.get_stride());
			Assert::AreEqual(1, pooling.get_filter_size());
			Assert::AreEqual((int)max_pooling, (int)pooling.get_pooling_fn());

			delete input;
		}
		TEST_METHOD(pooling_constructor_test_2)
		{
			matrix* input = create_matrix(6, 6, 2);
			pooling_layer pooling(input, 2, 2, average_pooling);
			Assert::AreEqual(3, pooling.get_activations().height);
			Assert::AreEqual(3, pooling.get_activations().width);
			Assert::AreEqual(2, pooling.get_activations().depth);

			Assert::IsTrue(input == pooling.get_input_p());
			Assert::AreEqual(2, pooling.get_stride());
			Assert::AreEqual(2, pooling.get_filter_size());
			Assert::AreEqual((int)average_pooling, (int)pooling.get_pooling_fn());

			delete input;
		}
		TEST_METHOD(pooling_constructor_test_invalid_arguments)
		{
			matrix* input = create_matrix(6, 6, 2);
			
			//wrong filter
			try
			{
				pooling_layer pooling(input, 0, 2, average_pooling);
				Assert::Fail();
			}
			catch (std::invalid_argument e)
			{
				Assert::AreEqual("filter size must be greater than 0", e.what());
			}

			//wrong stride
			try
			{
				pooling_layer pooling(input, 2, 0, average_pooling);
				Assert::Fail();
			}
			catch (std::invalid_argument e)
			{
				Assert::AreEqual("stride must be greater than 0", e.what());
			}

			delete input;
			input = nullptr;
			//wrong input
			try
			{
				pooling_layer pooling(input, 2, 2, average_pooling);
				Assert::Fail();
			}
			catch (std::invalid_argument e)
			{
				Assert::AreEqual("input cannot be null", e.what());
			}
		}
		TEST_METHOD(feed_forward_test_min_pooling)
		{
			matrix* input = create_matrix(6, 6, 2);
			set_all(*input, 1);
			set_at(*input, 0, 0, 0, 0);
			
			pooling_layer pooling(input, 2, 2, min_pooling);
			pooling.forward_propagation();
			Assert::AreEqual(3, pooling.get_activations().height);
			Assert::AreEqual(3, pooling.get_activations().width);
			Assert::AreEqual(2, pooling.get_activations().depth);

			Assert::AreEqual(0.0f, pooling.get_activations().data[0]);
			for (int i = 1; i < pooling.get_activations().data.size(); i++)
			{
				Assert::AreEqual(1.0f, pooling.get_activations().data[i]);
			}

			delete input;
		}
	};
}