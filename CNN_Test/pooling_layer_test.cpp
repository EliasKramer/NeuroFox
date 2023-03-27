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
			pooling_layer* pooling = create_pooling_layer(input , 1, 1, max_pooling);
			Assert::AreEqual(2, pooling->output.height);
			Assert::AreEqual(2, pooling->output.width);
			Assert::AreEqual(1, pooling->output.depth);

			Assert::IsTrue(input == pooling->input);
			Assert::AreEqual(1, pooling->stride);
			Assert::AreEqual(1, pooling->filter_size);
			Assert::AreEqual((int)max_pooling, (int)pooling->pooling_fn);

			delete pooling;
			delete input;
		}
		TEST_METHOD(pooling_constructor_test_2)
		{
			matrix* input = create_matrix(6, 6, 2);
			pooling_layer* pooling = create_pooling_layer(input, 2, 2, average_pooling);
			Assert::AreEqual(3, pooling->output.height);
			Assert::AreEqual(3, pooling->output.width);
			Assert::AreEqual(2, pooling->output.depth);

			Assert::IsTrue(input == pooling->input);
			Assert::AreEqual(2, pooling->stride);
			Assert::AreEqual(2, pooling->filter_size);
			Assert::AreEqual((int)average_pooling, (int)pooling->pooling_fn);

			delete pooling;
			delete input;
		}
		TEST_METHOD(pooling_constructor_test_invalid_arguments)
		{
			matrix* input = create_matrix(6, 6, 2);
			pooling_layer* pooling;
			
			//wrong filter
			try
			{
				pooling = create_pooling_layer(input, 0, 2, average_pooling);
				delete pooling;
				Assert::Fail();
			}
			catch (const char* msg)
			{
				Assert::AreEqual("filter size must be greater than 0", msg);
			}

			//wrong stride
			try
			{
				pooling = create_pooling_layer(input, 2, 0, average_pooling);
				delete pooling;
				Assert::Fail();
			}
			catch (const char* msg)
			{
				Assert::AreEqual("stride must be greater than 0", msg);
			}

			delete input;
			input = nullptr;
			//wrong input
			try
			{
				pooling = create_pooling_layer(input, 2, 2, average_pooling);
				Assert::Fail();
			}
			catch (const char* msg)
			{
				Assert::AreEqual("input cannot be null", msg);
			}
		}
		TEST_METHOD(feed_forward_test_min_pooling)
		{
			matrix* input = create_matrix(6, 6, 2);
			set_all(*input, 1);
			set_at(*input, 0, 0, 0, 0);
			
			pooling_layer* pooling = create_pooling_layer(input, 2, 2, min_pooling);
			feed_forward(*pooling);
			Assert::AreEqual(3, pooling->output.height);
			Assert::AreEqual(3, pooling->output.width);
			Assert::AreEqual(2, pooling->output.depth);

			Assert::AreEqual(0.0f, pooling->output.data[0]);
			for (int i = 1; i < pooling->output.data.size(); i++)
			{
				Assert::AreEqual(1.0f, pooling->output.data[i]);
			}

			delete pooling;
			delete input;
		}
	};
}