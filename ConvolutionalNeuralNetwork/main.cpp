#include <iostream>
#include "mnist_digit_overlord.hpp"

int main()
{
    mnist_digit_overlord overlord;

    std::cout << "start testing" << std::endl;
    test_result t_result = overlord.test();
    std::cout << "end testing" << std::endl;
    std::cout << t_result.to_string() << std::endl;

    return 0;
}