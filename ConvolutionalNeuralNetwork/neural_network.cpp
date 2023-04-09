#include "neural_network.hpp"
#include "util.hpp"

void neural_network::set_input(matrix* input)
{
	if (input == nullptr)
	{
		throw "Input is nullptr.";
	}

	if (input_format_set == false ||
		matrix_equal_format(input_format, *input) == false)
	{
		throw "Could not set Input. Input format is not set or does not match the input format.";
	}

	if (layers.empty())
	{
		throw "Could not set Input. No layers have been added yet.";
	}

	layers.front()->set_input(input);
}

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
	if (input_format_set == false)
		input_format_set = true;
	else
		throw std::runtime_error("Cannot set input format twice.");

	resize_matrix(this->input_format, given_input_format);
}

void neural_network::set_output_format(const matrix& given_output_format)
{
	if (output_format_set == false)
		output_format_set = true;
	else
		throw std::runtime_error("Cannot set output format twice.");

	resize_matrix(this->output_format, given_output_format);
	resize_matrix(this->cost_derivative, given_output_format);
}

const matrix& neural_network::get_output() const
{
	return *output_p;
}

void neural_network::add_layer(std::unique_ptr<layer>&& given_layer)
{
	//add the index of the layer to the vector of parameter layers
	//if the layer is not a pooling layer
	//because pooling layers do not have parameters
	if (given_layer->get_layer_type() != e_layer_type_t::pooling)
	{
		parameter_layer_indices.push_back(layers.size());
	}

	if (layers.empty())
	{
		//if there are no layers yet, the input format of the first layer
		//is the input format of the neural network
		given_layer.get()->set_input_format(input_format);
	}
	else
	{
		//if there are already layers,
		//set the previous layer of the new layer to the last layer
		given_layer.get()->set_previous_layer(*get_last_layer());
	}
	//putting the new layer into the vector of layers
	layers.push_back(std::move(given_layer));
}

void neural_network::add_fully_connected_layer(int num_neurons, e_activation_t activation_fn)
{
	std::unique_ptr<fully_connected_layer> new_layer =
		std::make_unique<fully_connected_layer>(num_neurons, activation_fn);

	add_layer(std::move(new_layer));
}

void neural_network::add_last_fully_connected_layer(e_activation_t activation_fn)
{
	std::unique_ptr<fully_connected_layer> new_layer =
		std::make_unique<fully_connected_layer>(output_format, activation_fn);

	add_layer(std::move(new_layer));
	output_p = get_last_layer()->get_activations_p();
}

void neural_network::add_convolutional_layer(int kernel_size, int number_of_kernels, int stride, e_activation_t activation_fn)
{
	/*
	//TODO check if the input format is correct
	std::unique_ptr<convolutional_layer> new_layer =
		std::make_unique<convolutional_layer>(
			input_for_new_layer,
			*get_last_layer_format(),
			kernel_size,
			number_of_kernels,
			stride,
			activation_fn);
	add_layer(std::move(new_layer));
	*/
}

void neural_network::add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type)
{
}

void neural_network::set_all_parameter(float value)
{
	for (auto& l : layers)
	{
		l->set_all_parameter(value);
	}
}

void neural_network::apply_noise(float range)
{
	for (auto& l : layers)
	{
		l->apply_noise(range);
	}
}

void neural_network::mutate(float range)
{
	if (parameter_layer_indices.empty())
	{
		throw std::runtime_error("Cannot mutate. No parameter layers have been added yet.");
	}
	int layer_idx = parameter_layer_indices[random_idx(parameter_layer_indices.size())];
	layers[layer_idx]->mutate(range);
}

float neural_network::test(std::vector<nn_data>& test_data)
{
	return 0.0f;
}

void neural_network::forward_propagation(matrix* input)
{
	set_input(input);
	//std::vector<std::unique_ptr<layer>>::iterator::value_type
	for (auto& l : layers)
	{
		l->forward_propagation();
	}
}

void neural_network::learn(std::vector<nn_data>& training_data)
{
}

void neural_network::back_propagation(nn_data* training_data)
{
	//checking for correct format
	if (!matrix_equal_format(training_data->get_label(), output_format))
	{
		throw std::runtime_error(
			"The expected output does not have the correct format.");
	}
	
	//feeding the data through
	forward_propagation(training_data->get_data_p());

	//calculating the cost derivative
	//calculate_cost_derivative(training_data->get_label_p());
	get_last_layer()->set_error_for_last_layer(training_data->get_label());

	//back propagating
	for (auto it = layers.rbegin(); it != layers.rend(); ++it)
	{
		(*it)->back_propagation();
	}

	//iterate over all parameter layers
	for (auto& l : parameter_layer_indices)
	{
		//update the parameters
		layers[l]->apply_deltas(1);
	}
}