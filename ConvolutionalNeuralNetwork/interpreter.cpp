#include "interpreter.hpp"

interpreter::interpreter()
	:input(nullptr)
{}

interpreter::interpreter(const matrix* given_input)
	: input(given_input) 
{}