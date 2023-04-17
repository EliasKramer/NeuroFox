#include "digit_interpreter.hpp"

digit_interpreter::digit_interpreter()
	: interpreter()
{}

void digit_interpreter::check_for_correct_input(const matrix* given_input) const
{
	if (given_input == nullptr)
		throw std::invalid_argument("Input is nullptr.");

	matrix required_input_format(1, 10, 1);

	if (matrix::equal_format(required_input_format, *given_input) == false)
	{
		throw std::invalid_argument("Input format is not correct.");
	}
}

digit_interpreter::digit_interpreter(const matrix* given_input)
	: interpreter(given_input)
{
	check_for_correct_input(given_input);
}

int digit_interpreter::get_interpretation() const
{
	return get_interpretation(input);
}

int digit_interpreter::get_interpretation(const matrix* given_input) const
{
	check_for_correct_input(given_input);

	float highest_activation = FLT_MIN;
	int highest_digit = 0;
	int curr_digit = 0;
	for each (const float curr_activation in given_input->flat_readonly())
	{
		if (curr_activation > highest_activation)
		{
			highest_activation = curr_activation;
			highest_digit = curr_digit;
		}
		curr_digit++;
	}

	return highest_digit;
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
	for each (const float curr_activation in given_input->flat_readonly())
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
	if(matrix::equal_format(a, b) == false)
		throw std::invalid_argument("Matrices are not the same format.");

	return get_interpretation(&a) == get_interpretation(&b);
}