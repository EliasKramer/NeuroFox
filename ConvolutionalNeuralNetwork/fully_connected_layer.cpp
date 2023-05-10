#include "fully_connected_layer.hpp"

float fully_connected_layer::get_weight_at(int input_layer_idx, int current_activation_idx) const
{
	return weights.get_at(input_layer_idx, current_activation_idx);
}

void fully_connected_layer::set_weight_at(int input_layer_idx, int current_activation_idx, float value)
{
	weights.set_at(input_layer_idx, current_activation_idx, value);
}

float fully_connected_layer::get_weight_delta_at(int input_layer_idx, int current_activation_idx) const
{
	return weight_deltas.get_at(input_layer_idx, current_activation_idx);
}

void fully_connected_layer::set_weight_delta_at(int input_layer_idx, int current_activation_idx, float value)
{
	weight_deltas.set_at(input_layer_idx, current_activation_idx, value);
}

float fully_connected_layer::get_passing_error_at(int input_layer_idx) const
{
	return passing_error->flat_readonly()[input_layer_idx];
}

void fully_connected_layer::set_passing_error_at(int input_layer_idx, float value)
{
	passing_error->flat()[input_layer_idx] = value;
}

void fully_connected_layer::forward_propagation_cpu()
{
	matrix::dot_product_flat(weights, *input, activations);
	matrix::add_flat(activations, biases, activations);
	activations.apply_activation_function(activation_fn);
}

void fully_connected_layer::back_propagation_cpu()
{
	if (!matrix::equal_format(activations, error))
	{
		throw std::invalid_argument("activations and error_right have different format");
	}

	for (int current_neuron_idx = 0; current_neuron_idx < activations.flat_readonly().size(); current_neuron_idx++)
	{
		float error_value = error.flat_readonly()[current_neuron_idx];
		//clear the error
		error.flat()[current_neuron_idx] = 0;

		float current_activation_value = activations.flat_readonly()[current_neuron_idx];
		float unactivated_activation = INVERSE[activation_fn](current_activation_value);
		float activation_derivative = DERIVATIVE[activation_fn](unactivated_activation);

		//bias change
		float bias_change = error_value * activation_derivative;
		bias_deltas.flat()[current_neuron_idx] += bias_change;

		//iterate input layer
		for (int current_input_idx = 0; current_input_idx < input->flat_readonly().size(); current_input_idx++)
		{
			float current_previous_activation = input->flat_readonly()[current_input_idx];

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

void fully_connected_layer::forward_propagation_gpu()
{
	if (!gpu_weights || !gpu_biases || !gpu_activations)
	{
		throw std::invalid_argument("gpu_weights, gpu_biases or gpu_activations is null");
	}

	gpu_dot_product(*gpu_weights.get(), *gpu_input, *gpu_activations.get());
	gpu_add(*gpu_activations.get(), *gpu_biases.get(), *gpu_activations.get());
	GPU_ACTIVATION[activation_fn](*gpu_activations.get());
}

void fully_connected_layer::back_propagation_gpu()
{
}

fully_connected_layer::fully_connected_layer(
	int number_of_neurons,
	e_activation_t activation_function
)
	:fully_connected_layer(
		matrix(1, number_of_neurons, 1),
		activation_function)
{}

fully_connected_layer::fully_connected_layer(
	const matrix& activation_format,
	e_activation_t activation_function
)
	:layer(e_layer_type_t::fully_connected)
{
	activations.resize(activation_format);

	//the error is needed for every output (we need it for learning)
	error.resize(activation_format);

	//weights and biases are the parameter values
	biases.resize(activation_format);
	weights.resize(0, 0, 0);

	//when training the network, we need to know how much the weights and biases need to change
	bias_deltas.resize(activation_format);

	weight_deltas.resize(0, 0, 0);

	//the activation function for making the output not linear
	activation_fn = activation_function;
}

void fully_connected_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);

	weights.resize(
		input_format.flat_readonly().size(),
		activations.flat_readonly().size(),
		1);
	weight_deltas.resize(weights);
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
	biases.set_all(value);
	weights.set_all(value);
}

void fully_connected_layer::apply_noise(float range)
{
	biases.apply_noise(range);
	weights.apply_noise(range);
}

void fully_connected_layer::mutate(float range)
{
	if (biased_coin_toss(
		(float)weights.flat_readonly().size(),
		(float)biases.flat_readonly().size()))
	{
		weights.mutate(range);
	}
	else
	{
		biases.mutate(range);
	}
}

void fully_connected_layer::apply_deltas(int number_of_inputs)
{
	for (int i = 0; i < biases.flat_readonly().size(); i++)
	{
		float current_bias = biases.flat()[i];
		float avg_bias_delta = bias_deltas.flat()[i] / number_of_inputs;
		biases.flat()[i] = current_bias - avg_bias_delta;
		bias_deltas.flat()[i] = 0;
	}

	for (int i = 0; i < weights.flat_readonly().size(); i++)
	{
		float current_weight = weights.flat()[i];
		float avg_weight_delta = weight_deltas.flat()[i] / number_of_inputs;
		weights.flat()[i] = current_weight - avg_weight_delta;
		weight_deltas.flat()[i] = 0;
	}
}

void fully_connected_layer::enable_gpu()
{
	layer::enable_gpu();
	
	gpu_weights = std::make_unique<gpu_matrix>(weights, true);
	gpu_biases = std::make_unique<gpu_matrix>(biases, true);
}

void fully_connected_layer::disable_gpu()
{
	layer::disable_gpu();

	gpu_biases = nullptr;
	gpu_weights = nullptr;
}
