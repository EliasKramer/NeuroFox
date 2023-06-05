#include "neural_network.hpp"
#include "util.hpp"

layer* neural_network::get_last_layer()
{
	//the last layer is the layer that was added last or nullptr 
	//if there are no layers yet
	return layers.empty() ? nullptr : layers.back().get();
}

neural_network::neural_network()
{}

neural_network::neural_network(const neural_network & source)
{
	//if same return
	if (this == &source)
		return;

	//copy all layers
	for (auto& curr : source.layers)
	{
		layers.push_back(std::move(curr->clone()));
	}

	//copy the input format
	input_format = source.input_format;

	//copy the parameter layer indices
	parameter_layer_indices = source.parameter_layer_indices;

	//copy the gpu_enabled flag
	gpu_enabled = source.gpu_enabled;
}

size_t neural_network::get_param_count() const
{
	size_t result = 0;

	for (int i = 0; i < parameter_layer_indices.size(); i++)
	{
		result += layers[parameter_layer_indices[i]].get()->get_parameter_count();
	}

	return result;
}

size_t neural_network::get_param_byte_size() const
{
	return get_param_count() * sizeof(float);
}

void neural_network::set_input_format(vector3 given_input_format)
{
	if (input_format.item_count() != 0)
		throw std::runtime_error("Cannot set input format twice.");

	this->input_format = given_input_format;
}

const matrix& neural_network::get_output_readonly() const
{
	if (layers.empty())
		throw std::runtime_error("Cannot get output of neural network with no layers.");
	return layers.back().get()->get_activations_readonly();
}

matrix& neural_network::get_output()
{
	if (layers.empty())
		throw std::runtime_error("Cannot get output of neural network with no layers.");
	return layers.back().get()->get_activations();
}

void neural_network::add_layer(std::unique_ptr<layer>&& given_layer)
{
	//add the index of the layer to the vector of parameter layers
	//if the layer is not a pooling layer
	//because pooling layers do not have parameters
	if (given_layer->get_layer_type() != e_layer_type_t::pooling)
	{
		parameter_layer_indices.push_back((int)layers.size());
	}

	if (layers.empty())
	{
		//if there are no layers yet, the input format of the first layer
		//is the input format of the neural network
		given_layer->set_input_format(input_format);
	}
	else
	{
		//if there are already layers,
		//set the previous layer of the new layer to the last layer
		given_layer->set_input_format(get_last_layer()->get_activations_readonly().get_format());
	}

	//putting the new layer into the vector of layers
	layers.push_back(std::move(given_layer));
}

float neural_network::calculate_cost(const matrix& expected_output)
{
	if (matrix::equal_format(get_output_readonly(), expected_output) == false)
	{
		throw std::runtime_error("Output format does not match expected output format.");
	}

	float cost = 0.0f;
	for (int i = 0; i < expected_output.item_count(); i++)
	{
		float expected = expected_output.get_at_flat_host(i);
		float actual = get_output_readonly().get_at_flat_host(i);
		cost += ((actual - expected) * (actual - expected));
	}
	return cost;
}

void neural_network::sync_device_and_host()
{
	if (gpu_enabled)
	{
		for (auto& l : parameter_layer_indices)
		{
			layers[l]->sync_device_and_host();
		}
	}
}

void neural_network::add_fully_connected_layer(int num_neurons, e_activation_t activation_fn)
{
	std::unique_ptr<fully_connected_layer> new_layer =
		std::make_unique<fully_connected_layer>(num_neurons, activation_fn);

	add_layer(std::move(new_layer));
}

void neural_network::add_fully_connected_layer(vector3 neuron_format, e_activation_t activation_fn)
{
	std::unique_ptr<fully_connected_layer> new_layer =
		std::make_unique<fully_connected_layer>(neuron_format, activation_fn);
	add_layer(std::move(new_layer));
}

void neural_network::add_convolutional_layer(
	int number_of_kernels,
	int kernel_size,
	int stride,
	e_activation_t activation_fn)
{
	std::unique_ptr<convolutional_layer> new_layer =
		std::make_unique<convolutional_layer>(
			number_of_kernels,
			kernel_size,
			stride,
			activation_fn);

	add_layer(std::move(new_layer));
}

void neural_network::add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type)
{
	//TODO
}

void neural_network::set_all_parameter(float value)
{
	//for parameter layers
	for (auto& l : parameter_layer_indices)
	{
		layers[l]->set_all_parameter(value);
	}
}

void neural_network::apply_noise(float range)
{
	//for parameter layers
	for (auto& l : parameter_layer_indices)
	{
		layers[l]->apply_noise(range);
	}
	sync_device_and_host();
}

void neural_network::mutate(float range)
{
	if (parameter_layer_indices.empty())
	{
		throw std::runtime_error("Cannot mutate. No parameter layers have been added yet.");
	}
	int layer_idx = parameter_layer_indices[random_idx((int)parameter_layer_indices.size())];
	layers[layer_idx]->mutate(range);

	if (gpu_enabled)
	{
		layers[layer_idx]->sync_device_and_host();
	}
}
/*
test_result neural_network::test(const std::vector<std::unique_ptr<nn_data>>& test_data)
{
	test_result result;
	result.data_count = test_data.size();
	int correct_predictions = 0;
	float cost_sum = 0.0f;
	auto start = std::chrono::high_resolution_clock::now();

	for (auto& curr_data : test_data)
	{
		forward_propagation_cpu(curr_data.get()->get_data());
		if (get_interpreter<interpreter>()->same_result(*get_output(), curr_data.get()->get_label()))
		{
			correct_predictions++;
		}
		cost_sum += calculate_cost(curr_data.get()->get_label());
	}

	auto end = std::chrono::high_resolution_clock::now();

	result.time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	result.accuracy = (float)correct_predictions / (float)result.data_count;
	result.avg_cost = cost_sum / (float)result.data_count;

	return result;
}*/

void neural_network::forward_propagation(const matrix& input)
{
	matrix* last_layer = nullptr;
	//std::vector<std::unique_ptr<layer>>::iterator::value_type
	for (auto& l : layers)
	{
		l->forward_propagation(
			last_layer == nullptr ?
			input :
			*last_layer
		);
		last_layer = l.get()->get_activations_p();
	}
}

void neural_network::back_propagation(const matrix& given_data, const matrix& given_label)
{
	//feeding the data through
	forward_propagation(given_data);

	//calculating the cost derivative
	//calculate_cost_derivative(training_data->get_label_p());
	get_last_layer()->set_error_for_last_layer_cpu(given_label);

	//we start from the last layer
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		const matrix& input =
			i == 0 ?
			given_data :
			layers[i - 1].get()->get_activations_readonly();

		matrix* passing_error =
			i == 0 ?
			nullptr :
			layers[i - 1].get()->get_error_p();

		layers[i].get()->back_propagation(input, passing_error);
	}
}

void neural_network::apply_deltas(size_t training_data_count, float learning_rate)
{
	//iterate over all parameter layers
	for (auto& l : parameter_layer_indices)
	{
		//update the parameters
		layers[l]->apply_deltas(training_data_count, learning_rate);
	}
}
void neural_network::enable_gpu_mode()
{
	int device_count = 0;
	cudaError_t error = cudaGetDeviceCount(&device_count);

	if (error != cudaSuccess)
	{
		throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
	}

	if (device_count == 0)
	{
		throw std::runtime_error("No CUDA capable devices (GPUs) found.");
	}

	for (auto& l : layers)
	{
		l->enable_gpu_mode();
	}

	gpu_enabled = true;

	sync_device_and_host();
}