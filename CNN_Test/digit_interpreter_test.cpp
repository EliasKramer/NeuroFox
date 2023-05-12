#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/digit_interpreter.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(digit_interpreter_test)
	{
	public:
		TEST_METHOD(test_string_interpretation)
		{
			matrix input(1,10,1);

			for (int i = 0; i < 10; i++)
			{
				input.set_at(0, i, 0, (float)i);
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
		TEST_METHOD(interpretion_test)
		{
			matrix input(1, 10, 1);

			input.set_all(0);

			input.set_at(0, 1, 0, .03f);

			digit_interpreter interpreter(&input);
			int output = interpreter.get_interpretation();

			Assert::AreEqual(1, output);
		}
		TEST_METHOD(test_same_result)
		{
			matrix a(1, 10, 1);
			matrix b(1, 10, 1);

			for (int i = 0; i < 10; i++)
			{
				a.set_at(0, i, 0, (float)i);
				b.set_at(0, i, 0, (float)i);
			}

			a.set_at(0, 1, 0, .03f);
			b.set_at(0, 1, 0, .03f);

			digit_interpreter interpreter(&a);
			bool output = interpreter.same_result(a, b);

			Assert::AreEqual(true, output);
		}
		TEST_METHOD(test_not_same_result)
		{
			matrix a(1, 10, 1);
			matrix b(1, 10, 1);
			a.set_all(0);
			b.set_all(0);

			a.set_at(0, 1, 0, .03f);

			b.set_at(0, 1, 0, .02f);
			b.set_at(0, 2, 0, .04f);

			digit_interpreter interpreter;
			bool output = interpreter.same_result(a, b);

			Assert::AreEqual(false, output);
		}
	};
}