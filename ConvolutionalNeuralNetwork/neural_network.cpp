#include "neural_network.hpp"

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

	this->input_p = input;
}

layer* neural_network::get_last_layer()
{
	//the last layer is the layer that was added last or nullptr 
	//if there are no layers yet
	return layers.empty() ? nullptr : layers.back().get();
}

matrix* neural_network::get_last_layer_output()
{
	//the input for the new layer is the output of the last layer
	//or the input of the neural network if there is no last layer
	return
		get_last_layer() == nullptr ?
		input_p :
		get_last_layer()->get_activations_p();
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
}

const matrix& neural_network::get_output() const
{
	return *output_p;
}

void neural_network::add_layer(std::unique_ptr<layer>&& given_layer)
{
	//the input for the new layer is the output of the last layer
	given_layer.get()->set_input(get_last_layer_output());
	//putting the new layer into the vector of layers
	layers.push_back(std::move(given_layer));
}

void neural_network::add_fully_connected_layer(int num_neurons, e_activation_t activation_fn)
{
	layer* last_layer = get_last_layer();
	matrix* input_for_new_layer = get_last_layer_output();

	//TODO check if the input format is correct

	std::unique_ptr<fully_connected_layer> new_layer = 
		std::make_unique<fully_connected_layer>(input_for_new_layer, num_neurons, activation_fn);

	add_layer(std::move(new_layer));
}

void neural_network::add_convolutional_layer(int kernel_size, int number_of_kernels, int stride, e_activation_t activation_fn)
{
}

void neural_network::add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type)
{
}

void neural_network::set_interpreter(std::unique_ptr<interpreter>&& given_interpreter)
{
	interpreter_p = std::move(given_interpreter);
}

const interpreter* neural_network::get_interpreter() const
{
	return interpreter_p.get();
}

void neural_network::forward_propagation(matrix* input)
{
	set_input(input);
	//std::vector<std::unique_ptr<layer>>::iterator::value_type
	for (auto& layer : layers)
	{
		layer->forward_propagation();
	}
}

void neural_network::back_propagation(const matrix& expected_output)
{
	//TODO
}