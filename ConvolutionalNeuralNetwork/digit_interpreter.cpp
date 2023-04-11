#include "digit_interpreter.hpp"

void digit_interpreter::check_for_correct_input(const matrix* given_input) const
{
	if (given_input == nullptr)
		throw std::invalid_argument("Input is nullptr.");

	matrix required_input_format;
	resize_matrix(required_input_format, 1, 10, 1);

	if (matrix_equal_format(required_input_format, *given_input) == false)
	{
		throw std::invalid_argument("Input format is not correct.");
	}
}

digit_interpreter::digit_interpreter(const matrix* given_input)
	: interpreter(given_input)
{
	check_for_correct_input(given_input);
}

std::string digit_interpreter::get_string_interpretation() const
{
	return get_string_interpretation(input);
}

std::string digit_interpreter::get_string_interpretation(const matrix* given_input) const
{
	check_for_correct_input(given_input);

	std::string output = "";
	int curr_digit = 0;

	float highest_activation = FLT_MIN;
	int highest_digit = 0;
	for each (const float curr_activation in matrix_flat_readonly(*given_input))
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

bool digit_interpreter::same_result(const matrix& a, const matrix& b) const
{
	if(matrix_equal_format(a, b) == false)
		throw std::invalid_argument("Matrices are not the same format.");

	int highest_activation_a = 0;
	int highest_activation_b = 0;

	float highest_activation_value_a = FLT_MIN;
	float highest_activation_value_b = FLT_MIN;

	for (int i = 0; i < a.data.size(); i++)
	{
		if (a.data[i] > highest_activation_value_a)
		{
			highest_activation_value_a = a.data[i];
			highest_activation_a = i;
		}

		if (b.data[i] > highest_activation_value_b)
		{
			highest_activation_value_b = b.data[i];
			highest_activation_b = i;
		}
	}

	return highest_activation_a == highest_activation_b;
}