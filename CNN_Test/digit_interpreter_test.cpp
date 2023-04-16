#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/digit_interpreter.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(digit_interpreter_test)
	{
	public:
		TEST_METHOD(testing_expected_behaviour)
		{
			matrix input(1,10,1);

			for (int i = 0; i < 10; i++)
			{
				input.set_at(0, i, 0, i);
			}

			input.set_at( 0, 4, 0, 11);

			digit_interpreter interpreter(&input);
			std::string output = interpreter.get_string_interpretation();

			Assert::AreEqual(
				std::string("0: 0.000000\n") +
				std::string("1: 1.000000\n") +
				std::string("2: 2.000000\n") +
				std::string("3: 3.000000\n") +
				std::string("4: 11.000000\n") +
				std::string("5: 5.000000\n") +
				std::string("6: 6.000000\n") +
				std::string("7: 7.000000\n") +
				std::string("8: 8.000000\n") +
				std::string("9: 9.000000\n") +
				std::string("result: 4\n"),
				output);
		}
	};
}
