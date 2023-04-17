#pragma once
#include "interpreter.hpp"

class digit_interpreter : public interpreter 
{
private:
	void check_for_correct_input(const matrix* given_input) const;
public:
	digit_interpreter();
	digit_interpreter(const matrix* given_input);
	
	int get_interpretation() const;
	int get_interpretation(const matrix* given_input) const;

	std::string get_string_interpretation() const override;
	std::string get_string_interpretation(const matrix* given_input) const override;

	bool same_result(const matrix& a, const matrix& b) const override;
};