#pragma once
#include "matrix.hpp"
class interpreter {
protected:
	const matrix& input;
public:
	interpreter(const matrix& given_input);

	virtual std::string get_string_interpretation() const = 0;
};