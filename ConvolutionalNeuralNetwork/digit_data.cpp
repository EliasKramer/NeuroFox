#include "digit_data.hpp"

digit_data::digit_data(const digit_image_t& data, const std::string& label)
{
	resize_matrix(this->data, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1);
	resize_matrix(this->label, 1, 10, 1);

	for (int y = 0; y < IMAGE_SIZE_Y; y++)
	{
		for (int x = 0; x < IMAGE_SIZE_X; x++)
		{
			set_at(this->data, x, y, 0, data.matrix[y][x]);
		}
	}

	set_all(this->label, 0);
	set_at(this->label, 0, std::stoi(label), 0, 1);
}

std::vector<digit_data> digit_data::get_digit_training_data(std::string path)
{
	digit_image_collection_t training_data_mnist = load_mnist_data(
		path + "/train-images.idx3-ubyte",
		path + "/train-labels.idx1-ubyte");

	std::vector<digit_data> training_data;
	for each (digit_image_t curr in training_data_mnist)
	{
		digit_data curr_data(curr, curr.label);
		training_data.push_back(curr_data);
	}

	return training_data;
}

std::vector<digit_data> digit_data::get_digit_testing_data(std::string path)
{
	digit_image_collection_t testing_data_mnist = load_mnist_data(
		path + "/t10k-images.idx3-ubyte",
		path + "/t10k-labels.idx1-ubyte");

	std::vector<digit_data> testing_data;
	for each (digit_image_t curr in testing_data_mnist)
	{
		digit_data curr_data(curr, curr.label);
		testing_data.push_back(curr_data);
	}

	return testing_data;
}