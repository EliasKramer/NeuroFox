#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/fully_connected_layer.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(FullyConnectedLayerTest)
	{
	public:
		TEST_METHOD(creating_fully_connected_layer_simple)
		{
			fully_connected_layer fc_layer(5, relu_fn);
			fc_layer.set_input_format(vector3(1, 1, 1));

			matrix output = fc_layer.get_activations_readonly();

			Assert::AreEqual((size_t)5, output.get_height());
		}
		TEST_METHOD(creating_fully_connected_layer_with_multiple_inputs)
		{
			fully_connected_layer fc_layer(5, relu_fn);
			fc_layer.set_input_format(vector3(1,4,1));

			Assert::AreEqual((size_t)5, fc_layer.get_activations_readonly().get_height());

			Assert::AreEqual((size_t)5, fc_layer.get_weights().get_height());
			Assert::AreEqual((size_t)4, fc_layer.get_weights().get_width());

			Assert::AreEqual((size_t)5, fc_layer.get_biases().get_height());
		}
		TEST_METHOD(simple_forward_propagating)
		{
			matrix input(vector3(1, 1, 1));
			input.set_at_flat(0, 2);
			fully_connected_layer fc_layer(1, relu_fn);
			fc_layer.set_input_format(input.get_format());

			//CONTINUE HERE
			fc_layer.get_weights_ref().set_at_flat(0, 3);
			fc_layer.get_biases_ref().set_at_flat(0, 1);

			fc_layer.forward_propagation(input);

			Assert::AreEqual(7.0f, fc_layer.get_activations_readonly().get_at_flat_host(0));
		}
		TEST_METHOD(propagating_forward_5node_input_3node_layer)
		{
			matrix input(vector3(1, 5, 1));

			input.set_at_flat(0, 2);
			input.set_at_flat(1, 3);
			input.set_at_flat(2, 4);
			input.set_at_flat(3, 5);
			input.set_at_flat(4, 6);

			fully_connected_layer fc_layer(3, relu_fn);
			fc_layer.set_input_format(input.get_format());

			fc_layer.get_weights_ref().set_all(2);
			fc_layer.get_biases_ref().set_all(1);

			fc_layer.get_weights_ref().set_at(vector3(0, 0), -1);

			fc_layer.forward_propagation(input);

			Assert::AreEqual(35.0f, fc_layer.get_activations_readonly().get_at_flat_host(0));
			Assert::AreEqual(41.0f, fc_layer.get_activations_readonly().get_at_flat_host(1));
			Assert::AreEqual(41.0f, fc_layer.get_activations_readonly().get_at_flat_host(2));
		}
	};
}