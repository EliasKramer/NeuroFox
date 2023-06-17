#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/math_functions.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(Math)
	{
	public:

		TEST_METHOD(sigmoid_test)
		{
			Assert::AreEqual(0.5f, sigmoid(0.0f));
			Assert::AreEqual(0.7310585786300049f, sigmoid(1.0f));
			Assert::AreEqual(0.2689414213699951f, sigmoid(-1.0f));
		}
		TEST_METHOD(relu_test)
		{
			Assert::AreEqual(0.0f, relu(0.0f));
			Assert::AreEqual(1.0f, relu(1.0f));
			Assert::AreEqual(0.0f, relu(-1.0f));
		}
	};
}
