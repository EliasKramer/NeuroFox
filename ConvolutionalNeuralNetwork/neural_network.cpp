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

void neural_network::set_input_format(const matrix& given_input_format)
{
	if (input_format.item_count() != 0)
		throw std::runtime_error("Cannot set input format twice.");

	this->input_format.resize(given_input_format);
}

const matrix& neural_network::get_output() const
{
	if(layers.empty())
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
		given_layer->set_input_format(get_last_layer()->get_activations());
	}

	//putting the new layer into the vector of layers
	layers.push_back(std::move(given_layer));
}

void neural_network::apply_deltas(int training_data_count)
{
	//iterate over all parameter layers
	for (auto& l : parameter_layer_indices)
	{
		//update the parameters
		layers[l]->apply_deltas(training_data_count);
	}
}

float neural_network::calculate_cost(const matrix& expected_output)
{
	if (matrix::equal_format(get_output(), expected_output) == false)
	{
		throw std::runtime_error("Output format does not match expected output format.");
	}

	float cost = 0.0f;
	for (int i = 0; i < expected_output.flat_readonly().size(); i++)
	{
		float expected = expected_output.flat_readonly()[i];
		float actual = get_output().flat_readonly()[i];
		cost += ((actual - expected) * (actual - expected));
	}
	return cost;
}

void neural_network::add_fully_connected_layer(int num_neurons, e_activation_t activation_fn)
{
	std::unique_ptr<fully_connected_layer> new_layer =
		std::make_unique<fully_connected_layer>(num_neurons, activation_fn);

	add_layer(std::move(new_layer));
}

void neural_network::add_fully_connected_layer(const matrix& neuron_format, e_activation_t activation_fn)
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
}

void neural_network::mutate(float range)
{
	if (parameter_layer_indices.empty())
	{
		throw std::runtime_error("Cannot mutate. No parameter layers have been added yet.");
	}
	int layer_idx = parameter_layer_indices[random_idx((int)parameter_layer_indices.size())];
	layers[layer_idx]->mutate(range);
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

void neural_network::forward_propagation_cpu(const matrix& input)
{
	matrix* last_layer = nullptr;
	//std::vector<std::unique_ptr<layer>>::iterator::value_type
	for (auto& l : layers)
	{
		l->forward_propagation_cpu(
			last_layer == nullptr ?
			input :
			*last_layer
		);
		last_layer = l.get()->get_activations_p();
	}
}

void neural_network::forward_propagation_gpu(const gpu_matrix& input)
{
	if (gpu_enabled == false)
	{
		enable_gpu();
	}


}

void neural_network::learn(
	const std::vector<std::unique_ptr<nn_data>>& training_data,
	int batch_size,
	int epochs)
{
	batch_handler batch(training_data, batch_size);

	for (int i = 0; i < epochs; i++)
	{
		//std::cout << "Epoch " << i << std::endl;
		int x = 0;
		for (auto curr_data = batch.get_batch_start(); curr_data != batch.get_batch_end(); ++curr_data)
		{
			learn_once(*curr_data, false);
			apply_deltas(batch_size);
		}
		batch.calculate_new_batch();
	}
}

void neural_network::learn_once(const std::unique_ptr<nn_data>& training_data, bool apply_changes)
{
	//feeding the data through
	forward_propagation_cpu(training_data.get()->get_data());

	//calculating the cost derivative
	//calculate_cost_derivative(training_data->get_label_p());
	get_last_layer()->set_error_for_last_layer_cpu(
		training_data.get()->get_label());

	//we start from the last layer
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		const matrix& input =
			i == 0 ?
			training_data.get()->get_data() :
			layers[i - 1].get()->get_activations();

		matrix* passing_error =
			i == 0 ?
			nullptr :
			layers[i - 1].get()->get_error_p();

			layers[i].get()->back_propagation_cpu(input, passing_error);
	}

	//if we only train on one training data piece
	//we can apply the changes right away
	if (apply_changes)
	{
		apply_deltas(1);
	}
}

void neural_network::enable_gpu()
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
		l->enable_gpu();
	}

	gpu_enabled = true;
}