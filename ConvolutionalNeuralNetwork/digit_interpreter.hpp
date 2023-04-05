#pragma once
#include "interpreter.hpp"

class digit_interpreter : public interpreter 
{
public:
	digit_interpreter(const matrix& given_input);

	std::string get_string_interpretation() const override;
};