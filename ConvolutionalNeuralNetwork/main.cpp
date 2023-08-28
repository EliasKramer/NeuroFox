#include <iostream>
#include "../models/mnist_digit_images/mnist_digit_overlord.hpp"

int main()
{

	mnist_digit_overlord overlord;

	for (int i = 0; ; i++)
	{
		std::cout << "Epoch: " << i << std::endl;

		overlord.train(1, 1, 0.0001f);

		//std::cout << "start testing" << std::endl;
		test_result t_result = overlord.test();
		//std::cout << "testing done" << std::endl;
		std::cout << t_result.to_string() << "\n\n";

	}
	return 0;

	matrix m;
	m.get_string();
}