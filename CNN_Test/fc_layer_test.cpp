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
			matrix* input = new matrix(1, 1, 1);
			fully_connected_layer fc_layer(5, relu_fn);
			fc_layer.set_input_format(*input);
			fc_layer.set_input(input);

			matrix output = fc_layer.get_activations();

			Assert::AreEqual(5, output.get_height());

			delete input;
		}
		TEST_METHOD(creating_fully_connected_layer_with_multiple_inputs)
		{
			matrix* input = new matrix(1, 4, 1);
			fully_connected_layer fc_layer(5, relu_fn);
			fc_layer.set_input_format(*input);
			fc_layer.set_input(input);

			Assert::AreEqual(5, fc_layer.get_activations().get_height());

			Assert::AreEqual(5, fc_layer.get_weights().get_height());
			Assert::AreEqual(4, fc_layer.get_weights().get_width());

			Assert::AreEqual(5, fc_layer.get_biases().get_height());

			delete input;
		}
		TEST_METHOD(simple_forward_propagating)
		{
			matrix* input = new matrix(1, 1, 1);
			input->flat()[0] = 2;
			fully_connected_layer fc_layer(1, relu_fn);
			fc_layer.set_input_format(*input);
			fc_layer.set_input(input);

			//CONTINUE HERE
			fc_layer.get_weights_ref().flat()[0] = 3;
			fc_layer.get_biases_ref().flat()[0] = 1;

			fc_layer.forward_propagation();

			Assert::AreEqual(7.0f, fc_layer.get_activations().flat_readonly()[0]);

			delete input;
		}
		TEST_METHOD(propagating_forward_5node_input_3node_layer)
		{
			matrix* input = new matrix(1, 5, 1);
			input->flat()[0] = 2;
			input->flat()[1] = 3;
			input->flat()[2] = 4;
			input->flat()[3] = 5;
			input->flat()[4] = 6;
			fully_connected_layer fc_layer(3, relu_fn);
			fc_layer.set_input_format(*input);
			fc_layer.set_input(input);

			fc_layer.get_weights_ref().set_all(2);
			fc_layer.get_biases_ref().set_all(1);

			fc_layer.get_weights_ref().set_at(0, 0, -1);

			fc_layer.forward_propagation();

			Assert::AreEqual(35.0f, fc_layer.get_activations().flat_readonly()[0]);
			Assert::AreEqual(41.0f, fc_layer.get_activations().flat_readonly()[1]);
			Assert::AreEqual(41.0f, fc_layer.get_activations().flat_readonly()[2]);

			delete input;
		}
	};
}