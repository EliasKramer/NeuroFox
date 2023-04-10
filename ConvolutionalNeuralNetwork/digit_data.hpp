#pragma once
#include "nn_data.hpp"
#include "digit_training_data.hpp"

class digit_data : public nn_data
{
private:
	static std::vector<std::unique_ptr<nn_data>>
		get_mnist_digit_data(std::string image_path, std::string label_path);

public:
	digit_data(const digit_image_t& data, const std::string& label);

	static std::vector<std::unique_ptr<nn_data>>
		get_digit_training_data(std::string path);
	
	static std::vector<std::unique_ptr<nn_data>>
		get_digit_testing_data(std::string path);
};