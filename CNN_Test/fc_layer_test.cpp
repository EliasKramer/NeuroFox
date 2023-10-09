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
			fc_layer.set_input_format(vector3(1, 4, 1));

			Assert::AreEqual((size_t)5, fc_layer.get_activations_readonly().get_height());

			Assert::AreEqual((size_t)5, fc_layer.get_weights().get_height());
			Assert::AreEqual((size_t)4, fc_layer.get_weights().get_width());

			Assert::AreEqual((size_t)5, fc_layer.get_biases().get_height());
		}
		TEST_METHOD(simple_forward_propagating)
		{
			matrix input(vector3(1, 1, 1));
			input.set_at_flat_host(0, 2);
			fully_connected_layer fc_layer(1, relu_fn);
			fc_layer.set_input_format(input.get_format());

			//CONTINUE HERE
			fc_layer.get_weights_ref().set_at_flat_host(0, 3);
			fc_layer.get_biases_ref().set_at_flat_host(0, 1);

			fc_layer.forward_propagation(input);

			Assert::AreEqual(7.0f, fc_layer.get_activations_readonly().get_at_flat_host(0));
		}
		TEST_METHOD(propagating_forward_5node_input_3node_layer)
		{
			matrix input(vector3(1, 5, 1));

			input.set_at_flat_host(0, 2);
			input.set_at_flat_host(1, 3);
			input.set_at_flat_host(2, 4);
			input.set_at_flat_host(3, 5);
			input.set_at_flat_host(4, 6);

			fully_connected_layer fc_layer(3, relu_fn);
			fc_layer.set_input_format(input.get_format());

			fc_layer.get_weights_ref().set_all(2);
			fc_layer.get_biases_ref().set_all(1);

			fc_layer.get_weights_ref().set_at_host(vector3(0, 0), -1);

			fc_layer.forward_propagation(input);

			Assert::AreEqual(35.0f, fc_layer.get_activations_readonly().get_at_flat_host(0));
			Assert::AreEqual(41.0f, fc_layer.get_activations_readonly().get_at_flat_host(1));
			Assert::AreEqual(41.0f, fc_layer.get_activations_readonly().get_at_flat_host(2));
		}
		TEST_METHOD(partial_forward_prop_test_1)
		{
			matrix input(vector3(1, 2, 1));
			input.set_at_flat_host(0, 2);
			input.set_at_flat_host(1, 3);

			fully_connected_layer fc_layer(2, leaky_relu_fn);
			fc_layer.set_input_format(input.get_format());
			fc_layer.get_weights_ref().set_at_flat_host(0, 0);
			fc_layer.get_weights_ref().set_at_flat_host(1, 1);
			fc_layer.get_weights_ref().set_at_flat_host(2, 2);
			fc_layer.get_weights_ref().set_at_flat_host(3, 3);
			fc_layer.get_biases_ref().set_all(1.3);
			fc_layer.forward_propagation(input);
			matrix normal_forward = fc_layer.get_activations();
			fc_layer.get_activations_p()->set_all(0);

			matrix input_prev(vector3(1, 2, 1));
			input_prev.set_at_flat_host(0, 2);
			input_prev.set_at_flat_host(1, 2);
			matrix input_part(vector3(1, 2, 1));
			input_part.set_at_flat_host(0, 2);
			input_part.set_at_flat_host(1, 3);

			fc_layer.forward_propagation(input_prev);

			fc_layer.partial_forward_prop(input_part, input_prev, vector3(0, 1, 0));
			matrix partial_forward = fc_layer.get_activations();
			
			Assert::AreEqual(normal_forward.get_at_flat_host(0), partial_forward.get_at_flat_host(0));
			Assert::AreEqual(normal_forward.get_at_flat_host(1), partial_forward.get_at_flat_host(1));
		}
		TEST_METHOD(partial_forward_prop_test_2)
		{
			matrix input(vector3(1, 5, 1));
			input.set_all(1.5f);
			input.set_at_flat_host(0, 2);
			input.set_at_flat_host(1, 3);

			fully_connected_layer fc_layer(4, leaky_relu_fn);
			fc_layer.set_input_format(input.get_format());
			fc_layer.set_all_parameters(13.0123f);
			
			fc_layer.apply_noise(100);
			fc_layer.forward_propagation(input);
			matrix normal_forward = fc_layer.get_activations();
			fc_layer.get_activations_p()->set_all(0);

			matrix input_prev(vector3(1, 5, 1));
			input_prev.set_all(1.5f);
			input_prev.set_at_flat_host(0, 2);
			input_prev.set_at_flat_host(1, 2);
			matrix input_part(vector3(1, 5, 1));
			input_part.set_all(1.5f);
			input_part.set_at_flat_host(0, 2);
			input_part.set_at_flat_host(1, 3);

			std::string normal_forward_str = normal_forward.get_string();
			fc_layer.forward_propagation(input_prev);
			std::string part_prv_str = fc_layer.get_activations().get_string();

			fc_layer.partial_forward_prop(input_part, input_prev, vector3(0, 1, 0));
			matrix partial_forward = fc_layer.get_activations();

			std::string partial_forward_str = partial_forward.get_string();

			Assert::IsTrue(matrix::are_equal(normal_forward,partial_forward, 0.0001f));
		}
	};
}