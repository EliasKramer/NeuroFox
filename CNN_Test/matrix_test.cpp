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
			matrix* m = create_matrix(2, 3, 4);
			Assert::AreEqual(2, m->width);
			Assert::AreEqual(3, m->height);
			Assert::AreEqual(4, m->depth);
			Assert::AreEqual(24, (int)m->data.size());
		}
		TEST_METHOD(setting_getting)
		{
			matrix* m = create_matrix(2, 3, 4);
			set_at(*m, 0, 0, 0, 1);
			set_at(*m, 1, 0, 0, 2);
			set_at(*m, 0, 1, 0, 3);
			set_at(*m, 1, 1, 0, 4);
			set_at(*m, 0, 2, 0, 5);
			set_at(*m, 1, 2, 0, 6);
			set_at(*m, 0, 0, 1, 7);
			set_at(*m, 1, 0, 1, 8);
			set_at(*m, 0, 1, 1, 9);
			set_at(*m, 1, 1, 1, 10);
			set_at(*m, 0, 2, 1, 11);
			set_at(*m, 1, 2, 1, 12);
			set_at(*m, 0, 0, 2, 13);
			set_at(*m, 1, 0, 2, 14);
			set_at(*m, 0, 1, 2, 15);
			set_at(*m, 1, 1, 2, 16);
			set_at(*m, 0, 2, 2, 17);
			set_at(*m, 1, 2, 2, 18);
			set_at(*m, 0, 0, 3, 19);
			set_at(*m, 1, 0, 3, 20);
			set_at(*m, 0, 1, 3, 21);
			set_at(*m, 1, 1, 3, 22);
			set_at(*m, 0, 2, 3, 23);
			set_at(*m, 1, 2, 3, 24);
			Assert::AreEqual(1, (int)get_at(*m, 0, 0, 0));
			Assert::AreEqual(2, (int)get_at(*m, 1, 0, 0));
			Assert::AreEqual(3, (int)get_at(*m, 0, 1, 0));
			Assert::AreEqual(4, (int)get_at(*m, 1, 1, 0));
			Assert::AreEqual(5, (int)get_at(*m, 0, 2, 0));
			Assert::AreEqual(6, (int)get_at(*m, 1, 2, 0));
			Assert::AreEqual(7, (int)get_at(*m, 0, 0, 1));
			Assert::AreEqual(8, (int)get_at(*m, 1, 0, 1));
			Assert::AreEqual(9, (int)get_at(*m, 0, 1, 1));
			Assert::AreEqual(10, (int)get_at(*m, 1, 1, 1));
			Assert::AreEqual(11, (int)get_at(*m, 0, 2, 1));
			Assert::AreEqual(12, (int)get_at(*m, 1, 2, 1));
			Assert::AreEqual(13, (int)get_at(*m, 0, 0, 2));
			Assert::AreEqual(14, (int)get_at(*m, 1, 0, 2));
			Assert::AreEqual(15, (int)get_at(*m, 0, 1, 2));
			Assert::AreEqual(16, (int)get_at(*m, 1, 1, 2));
			Assert::AreEqual(17, (int)get_at(*m, 0, 2, 2));
			Assert::AreEqual(18, (int)get_at(*m, 1, 2, 2));
			Assert::AreEqual(19, (int)get_at(*m, 0, 0, 3));
			Assert::AreEqual(20, (int)get_at(*m, 1, 0, 3));
			Assert::AreEqual(21, (int)get_at(*m, 0, 1, 3));
			Assert::AreEqual(22, (int)get_at(*m, 1, 1, 3));
			Assert::AreEqual(23, (int)get_at(*m, 0, 2, 3));
			Assert::AreEqual(24, (int)get_at(*m, 1, 2, 3));
		}
		TEST_METHOD(dot_test)
		{
			matrix* m1 = create_matrix(2, 5, 1);
			matrix* m2 = create_matrix(3, 2, 1);

			set_all(*m1, 3);
			set_all(*m2, 2);

			matrix* m3 = create_matrix(2, 2, 1);
			dot(*m1, *m2, *m3);

			for (int i = 0; i < m3->data.size(); i++)
			{
				Assert::AreEqual(12, (int)m3->data[i]);
			}
		}
	};
}