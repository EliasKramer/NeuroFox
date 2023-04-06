#include "digit_data.hpp"

digit_data::digit_data(const digit_image_t& data, const std::string& label)
{
}

std::vector<digit_data> digit_data::get_digit_training_data()
{
	digit_image_collection_t training_data_mnist = load_mnist_data(
		"/data/digit_recognition/train-images-idx3-ubyte",
		"/data/digit_recognition/train-labels-idx1-ubyte");

	std::vector<digit_data> training_data;
	return training_data;
}

std::vector<digit_data> digit_data::get_digit_testing_data()
{
	digit_image_collection_t testing_data_mnist = load_mnist_data(
		"/data/digit_recognition/t10k-images-idx3-ubyte",
		"/data/digit_recognition/t10k-labels-idx1-ubyte");

	std::vector<digit_data> testing_data;
	return testing_data;
}