#include "digit_interpreter.hpp"

digit_interpreter::digit_interpreter(const matrix& given_input)
	: interpreter(given_input)
{
	matrix required_input_format;
	resize_matrix(required_input_format, 1, 10, 1);

	if (matrix_equal_format(required_input_format, given_input) == false)
	{
		throw std::invalid_argument("Input format is not correct.");
	}
}

std::string digit_interpreter::get_string_interpretation() const
{
	std::string output = "";
	int curr_digit = 1;

	float highest_activation = FLT_MIN;
	int highest_digit = 1;
	for each (const float curr_activation in matrix_flat_readonly(input))
	{
		if (curr_activation > highest_activation)
		{
			highest_activation = curr_activation;
			highest_digit = curr_digit;
		}

		output += std::to_string(curr_digit) + ": " + std::to_string(curr_activation) + "\n";
		curr_digit++;
	}
	output += "result: " + std::to_string(highest_digit) + "\n";

	return output;
}
