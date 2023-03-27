#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/neural_network.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(pasting_test)
	{
	public:
		
		TEST_METHOD(placeholder)
		{
			Assert::AreEqual(1, 1);
		}
	};
}
