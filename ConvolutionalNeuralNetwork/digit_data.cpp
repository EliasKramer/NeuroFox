#include "digit_data.hpp"

std::vector<std::unique_ptr<nn_data>>
digit_data::get_mnist_digit_data(std::string image_path, std::string label_path)
{
	digit_image_collection_t training_data_mnist = load_mnist_data(
		image_path,
		label_path);

	std::vector<std::unique_ptr<nn_data>> training_data;
	for each (digit_image_t curr in training_data_mnist)
	{
		std::unique_ptr<digit_data> curr_data = 
			std::make_unique<digit_data>(curr, curr.label);
		
		training_data.push_back(std::move(curr_data));
	}

	return training_data;
}

digit_data::digit_data(const digit_image_t& data, const std::string& label)
{
	this->data.initialize_format(IMAGE_SIZE_X, IMAGE_SIZE_Y, 1);
	this->label.initialize_format(1, 10, 1);

	for (int y = 0; y < IMAGE_SIZE_Y; y++)
	{
		for (int x = 0; x < IMAGE_SIZE_X; x++)
		{
			this->data.set_at(x, y, 0, data.image_matrix[y][x]);
		}
	}
	this->label.set_all(0);
	int_label = std::stoi(label);
	this->label.set_at(0, int_label, 0, 1);
}

const int digit_data::get_int_label() const
{
	return int_label;
}

std::vector<std::unique_ptr<nn_data>>
digit_data::get_digit_training_data(std::string path)
{
	return get_mnist_digit_data(
		path + "/train-images.idx3-ubyte",
		path + "/train-labels.idx1-ubyte");
}

std::vector<std::unique_ptr<nn_data>>
digit_data::get_digit_testing_data(std::string path)
{
	return get_mnist_digit_data(
		path + "/t10k-images.idx3-ubyte",
		path + "/t10k-labels.idx1-ubyte");
}