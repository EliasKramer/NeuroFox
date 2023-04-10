#include "fully_connected_layer.hpp"

float fully_connected_layer::get_weight_at(int input_layer_idx, int current_activation_idx) const
{
	return matrix_get_at(weights, input_layer_idx, current_activation_idx);
}

void fully_connected_layer::set_weight_at(int input_layer_idx, int current_activation_idx, float value)
{
	set_at(weights, input_layer_idx, current_activation_idx, value);
}

float fully_connected_layer::get_weight_delta_at(int input_layer_idx, int current_activation_idx) const
{
	return matrix_get_at(weight_deltas, input_layer_idx, current_activation_idx);
}

void fully_connected_layer::set_weight_delta_at(int input_layer_idx, int current_activation_idx, float value)
{
	set_at(weight_deltas, input_layer_idx, current_activation_idx, value);
}

float fully_connected_layer::get_passing_error_at(int input_layer_idx) const
{
	return passing_error->data[input_layer_idx];
}

void fully_connected_layer::set_passing_error_at(int input_layer_idx, float value)
{
	passing_error->data[input_layer_idx] = value;
}

fully_connected_layer::fully_connected_layer(
	int number_of_neurons,
	e_activation_t activation_function
)
	:fully_connected_layer(
		get_matrix(1, number_of_neurons, 1),
		activation_function)
{}

fully_connected_layer::fully_connected_layer(
	const matrix& activation_format,
	e_activation_t activation_function
)
	:layer(e_layer_type_t::fully_connected)
{
	resize_matrix(activations, activation_format);

	//the error is needed for every output (we need it for learning)
	resize_matrix(error, activation_format);

	//weights and biases are the parameter values
	resize_matrix(biases, activation_format);
	resize_matrix(weights, 0, 0, 0);

	//when training the network, we need to know how much the weights and biases need to change
	resize_matrix(bias_deltas, activation_format);
	resize_matrix(weight_deltas, 0, 0, 0);

	//the activation function for making the output not linear
	activation_fn = activation_function;
}

void fully_connected_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);

	resize_matrix(weights, input_format.data.size(), activations.data.size(), 1);
	resize_matrix(weight_deltas, weights);
}

const matrix& fully_connected_layer::get_weights() const
{
	return weights;
}

const matrix& fully_connected_layer::get_biases() const
{
	return biases;
}

matrix& fully_connected_layer::get_weights_ref()
{
	return weights;
}

matrix& fully_connected_layer::get_biases_ref()
{
	return biases;
}

void fully_connected_layer::set_all_parameter(float value)
{
	set_all(biases, value);
	set_all(weights, value);
}

void fully_connected_layer::apply_noise(float range)
{
	matrix_apply_noise(biases, range);
	matrix_apply_noise(weights, range);
}

void fully_connected_layer::mutate(float range)
{
	if (biased_coin_toss(weights.data.size(), biases.data.size()))
	{
		matrix_mutate(weights, range);
	}
	else
	{
		matrix_mutate(biases, range);
	}
}

void fully_connected_layer::forward_propagation()
{
	//TODO - straighten out the input matrix
	matrix_dot_flat(weights, *input, activations);
	matrix_add_flat(activations, biases, activations);
	matrix_apply_activation(activations, activation_fn);
}

void fully_connected_layer::back_propagation()
{
	if (!matrix_equal_format(activations, error))
	{
		throw std::invalid_argument("activations and error_right have different format");
	}

	for (int current_neuron_idx = 0; current_neuron_idx < activations.data.size(); current_neuron_idx++)
	{
		float error_value = matrix_flat_readonly(error)[current_neuron_idx];
		//clear the error
		matrix_flat(error)[current_neuron_idx] = 0;

		//TODO
		float current_activation_value = matrix_flat_readonly(activations)[current_neuron_idx];
		float unactivated_activation = INVERSE[activation_fn](current_activation_value);
		float activation_derivative = DERIVATIVE[activation_fn](unactivated_activation);

		//bias change
		float bias_change = error_value * activation_derivative;
		matrix_flat(bias_deltas)[current_neuron_idx] += bias_change;

		//iterate input layer
		for (int current_input_idx = 0; current_input_idx < matrix_flat_readonly(*input).size(); current_input_idx++)
		{
			float current_previous_activation = matrix_flat_readonly(*input)[current_input_idx];

			//this weight connects the current input node to the current neuron
			float current_weight = get_weight_at(current_input_idx, current_neuron_idx);

			float weight_change = error_value * activation_derivative * current_previous_activation;
			float new_passing_error = error_value * activation_derivative * current_weight;

			float current_weight_change = get_weight_delta_at(current_input_idx, current_neuron_idx);
			set_weight_delta_at(current_input_idx, current_neuron_idx, current_weight_change + weight_change);

			//passing error is null when this is the first layer
			if (passing_error != nullptr)
			{
				float current_error = get_passing_error_at(current_input_idx);
				set_passing_error_at(current_input_idx, current_error + new_passing_error);
			}
		}
	}
}

void fully_connected_layer::apply_deltas(int number_of_inputs)
{
	for (int i = 0; i < biases.data.size(); i++)
	{
		float current_bias = matrix_flat(biases)[i];
		float current_bias_delta = matrix_flat(bias_deltas)[i];
		matrix_flat(biases)[i] = current_bias - current_bias_delta / number_of_inputs;
		matrix_flat(bias_deltas)[i] = 0;
	}

	for (int i = 0; i < weights.data.size(); i++)
	{
		float current_weight = matrix_flat(weights)[i];
		float current_weight_delta = matrix_flat(weight_deltas)[i];
		matrix_flat(weights)[i] = current_weight - current_weight_delta / number_of_inputs;
		matrix_flat(weight_deltas)[i] = 0;
	}
}
