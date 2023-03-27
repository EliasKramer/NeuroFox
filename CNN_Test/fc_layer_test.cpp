#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/fully_connected_layer.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(FullyConnectedLayerTest)
	{
	public:
		TEST_METHOD(creating_fully_connected_layer_simple)
		{
			matrix* input = create_matrix(1, 1, 1);
			fully_connected_layer* fc_layer = create_fully_connected_layer(5, input, relu_fn);

			Assert::AreEqual(5, fc_layer->output.height);

			delete fc_layer;
			delete input;
		}
		TEST_METHOD(creating_fully_connected_layer_with_multiple_inputs)
		{
			matrix* input = create_matrix(1, 4, 1);
			fully_connected_layer* fc_layer = create_fully_connected_layer(5, input, relu_fn);

			Assert::AreEqual(5, fc_layer->output.height);
			Assert::AreEqual(5, fc_layer->weights.height);
			Assert::AreEqual(4, fc_layer->weights.width);

			Assert::AreEqual(5, fc_layer->biases.height);

			delete fc_layer;
			delete input;
		}
		TEST_METHOD(input_is_no_vector)
		{
			matrix* input = create_matrix(1, 4, 4);
			try
			{
				fully_connected_layer* fc_layer = create_fully_connected_layer(5, input, relu_fn);
			}
			catch (const char* msg)
			{
				Assert::AreEqual("Input matrix must be a vector (width and depth must be 1)", msg);
			}
			delete input;

			input = create_matrix(4, 1, 1);
			try
			{
				fully_connected_layer* fc_layer = create_fully_connected_layer(5, input, relu_fn);
			}
			catch (const char* msg)
			{
				Assert::AreEqual("Input matrix must be a vector (width and depth must be 1)", msg);
			}
			delete input;
		}
		TEST_METHOD(simple_feeding_forward)
		{
			matrix* input = create_matrix(1, 1, 1);
			input->data[0] = 2;
			fully_connected_layer* fc_layer = create_fully_connected_layer(1, input, relu_fn);
			fc_layer->weights.data[0] = 3;
			fc_layer->biases.data[0] = 1;

			feed_forward(*fc_layer);

			Assert::AreEqual(7.0f, fc_layer->output.data[0]);

			delete fc_layer;
			delete input;
		}
		TEST_METHOD(feeding_forward_5node_input_3node_layer)
		{
			matrix* input = create_matrix(1, 5, 1);
			input->data[0] = 2;
			input->data[1] = 3;
			input->data[2] = 4;
			input->data[3] = 5;
			input->data[4] = 6;
			fully_connected_layer* fc_layer = create_fully_connected_layer(3, input, relu_fn);
			set_all(fc_layer->weights, 2);
			set_all(fc_layer->biases, 1);

			set_at(fc_layer->weights, 0, 0, -1);

			feed_forward(*fc_layer);

			Assert::AreEqual(35.0f, fc_layer->output.data[0]);
			Assert::AreEqual(41.0f, fc_layer->output.data[1]);
			Assert::AreEqual(41.0f, fc_layer->output.data[2]);

			delete fc_layer;
			delete input;
		}
	};
}
