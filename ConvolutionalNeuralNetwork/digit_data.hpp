#pragma once
#include "nn_data.hpp"
#include "digit_training_data.hpp"

class digit_data : public nn_data
{
public:
	digit_data(const digit_image_t& data, const std::string& label);

	static std::vector<digit_data> get_digit_training_data(std::string path);
	static std::vector<digit_data> get_digit_testing_data(std::string path);
};