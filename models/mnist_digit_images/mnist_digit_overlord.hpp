#pragma once
#include "../../ConvolutionalNeuralNetwork/code/data_space.hpp"
#include "../../ConvolutionalNeuralNetwork/code/neural_network.hpp"
class mnist_digit_overlord
{
private:
	data_space ds_training;
	data_space ds_test;
	neural_network nn;

	void label_to_matrix(unsigned char label, matrix& m) const;
	float get_digit_cost(const matrix& output, const matrix& label) const;
	void print_digit_image(const matrix& m) const;
	void load_data(
		data_space& ds,
		std::string data_path,
		std::string label_path);

	//returns the flat index of the float with the highest value in the matrix
	size_t idx_of_max(const matrix& m) const;

	void enable_gpu();

public:
	mnist_digit_overlord();

	void debug_function();

	void save_to_file();
	void load_from_file();

	void print_nn_size() const;

	test_result test();
	test_result test_on_training_data();

	void train(size_t epochs, size_t batch_size, float learning_rate);
};