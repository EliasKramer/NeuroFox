#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/matrix.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(matrix_test)
	{
	public:

		TEST_METHOD(creating_matrix)
		{
			matrix* m = create_matrix(2, 2, 2);
			Assert::AreEqual(2, m->width);
			Assert::AreEqual(2, m->height);
			Assert::AreEqual(2, m->depth);
			Assert::AreEqual(8, (int)m->data.size());
		}
	};
}
