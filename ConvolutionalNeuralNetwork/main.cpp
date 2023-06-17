#include <iostream>
#include "../models/mnist_digit_images/mnist_digit_overlord.hpp"

int main()
{
	mnist_digit_overlord overlord;
	//overlord.debug_function();
	
	overlord.save_to_file();

	std::cout << "start testing" << std::endl;
	test_result t_result = overlord.test();
	std::cout << "end testing" << std::endl;
	std::cout << t_result.to_string() << std::endl;

	overlord.load_from_file();
	
	std::cout << "start testing after file loading" << std::endl;
	t_result = overlord.test();
	std::cout << "end testing after file loading" << std::endl;
	std::cout << t_result.to_string() << std::endl;
	
	return 0;
}