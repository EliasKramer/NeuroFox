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

			delete m;
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
			Assert::AreEqual(1, (int)matrix_get_at(*m, 0, 0, 0));
			Assert::AreEqual(2, (int)matrix_get_at(*m, 1, 0, 0));
			Assert::AreEqual(3, (int)matrix_get_at(*m, 0, 1, 0));
			Assert::AreEqual(4, (int)matrix_get_at(*m, 1, 1, 0));
			Assert::AreEqual(5, (int)matrix_get_at(*m, 0, 2, 0));
			Assert::AreEqual(6, (int)matrix_get_at(*m, 1, 2, 0));
			Assert::AreEqual(7, (int)matrix_get_at(*m, 0, 0, 1));
			Assert::AreEqual(8, (int)matrix_get_at(*m, 1, 0, 1));
			Assert::AreEqual(9, (int)matrix_get_at(*m, 0, 1, 1));
			Assert::AreEqual(10, (int)matrix_get_at(*m, 1, 1, 1));
			Assert::AreEqual(11, (int)matrix_get_at(*m, 0, 2, 1));
			Assert::AreEqual(12, (int)matrix_get_at(*m, 1, 2, 1));
			Assert::AreEqual(13, (int)matrix_get_at(*m, 0, 0, 2));
			Assert::AreEqual(14, (int)matrix_get_at(*m, 1, 0, 2));
			Assert::AreEqual(15, (int)matrix_get_at(*m, 0, 1, 2));
			Assert::AreEqual(16, (int)matrix_get_at(*m, 1, 1, 2));
			Assert::AreEqual(17, (int)matrix_get_at(*m, 0, 2, 2));
			Assert::AreEqual(18, (int)matrix_get_at(*m, 1, 2, 2));
			Assert::AreEqual(19, (int)matrix_get_at(*m, 0, 0, 3));
			Assert::AreEqual(20, (int)matrix_get_at(*m, 1, 0, 3));
			Assert::AreEqual(21, (int)matrix_get_at(*m, 0, 1, 3));
			Assert::AreEqual(22, (int)matrix_get_at(*m, 1, 1, 3));
			Assert::AreEqual(23, (int)matrix_get_at(*m, 0, 2, 3));
			Assert::AreEqual(24, (int)matrix_get_at(*m, 1, 2, 3));

			delete m;
		}
		TEST_METHOD(equal_test)
		{
			matrix* original_matrix = create_matrix(2, 5, 3);
			matrix* equal_matrix = create_matrix(2, 5, 3);
			matrix* one_different = create_matrix(2, 5, 3);
			matrix* size_different = create_matrix(2, 5, 2);

			set_all(*original_matrix, 1);
			set_all(*equal_matrix, 1);
			set_all(*one_different, 1);
			set_all(*size_different, 1);

			set_at(*one_different, 0, 0, 0, 2);

			Assert::IsTrue(are_equal(*original_matrix, *original_matrix));

			Assert::IsTrue(are_equal(*original_matrix, *equal_matrix));
			Assert::IsTrue(are_equal(*equal_matrix, *original_matrix));
			
			Assert::IsFalse(are_equal(*original_matrix, *one_different));
			Assert::IsFalse(are_equal(*one_different, *original_matrix));

			Assert::IsFalse(are_equal(*original_matrix, *size_different));
			Assert::IsFalse(are_equal(*size_different, *original_matrix));

			delete original_matrix;
			delete equal_matrix;
			delete one_different;
			delete size_different;
		}
		TEST_METHOD(dot_test_2D)
		{
			matrix* m1 = create_matrix(2, 4, 1);
			matrix* m2 = create_matrix(3, 2, 1);

			set_all(*m1, 3);
			set_all(*m2, 2);

			matrix* m3 = create_matrix(3, 4, 1);
			matrix_dot(*m1, *m2, *m3);

			for (int i = 0; i < m3->data.size(); i++)
			{
				Assert::AreEqual(12, (int)m3->data[i]);
			}

			delete m1;
			delete m2;
		}
		TEST_METHOD(dot_test_3D)
		{
			matrix* m1 = create_matrix(2, 4, 2);
			matrix* m2 = create_matrix(3, 2, 2);

			set_all(*m1, 3);
			set_all(*m2, 2);

			matrix* m3 = create_matrix(3, 4, 2);
			matrix_dot(*m1, *m2, *m3);

			for (int i = 0; i < m3->data.size(); i++)
			{
				Assert::AreEqual(12, (int)m3->data[i]);
			}

			delete m1;
			delete m2;
		}
		TEST_METHOD(hash_test)
		{
			// Test 1: Two identical matrices
			matrix* m1 = create_matrix(2, 4, 2);
			matrix* m2 = create_matrix(2, 4, 2);
			set_all(*m1, 3);
			set_all(*m2, 3);
			Assert::AreEqual(matrix_hash(*m1), matrix_hash(*m2));
			delete m1;
			delete m2;

			// Test 2: Two matrices with one different value
			m1 = create_matrix(2, 4, 2);
			m2 = create_matrix(2, 4, 2);
			set_all(*m1, 3);
			set_all(*m2, 3);
			set_at(*m2, 0, 0, 0, 4);
			Assert::AreNotEqual(matrix_hash(*m1), matrix_hash(*m2));
			delete m1;
			delete m2;

			// Test 3: Two matrices with different sizes
			m1 = create_matrix(2, 4, 2);
			m2 = create_matrix(2, 3, 2);
			set_all(*m1, 0);
			set_all(*m2, 0);
			Assert::AreNotEqual(matrix_hash(*m1), matrix_hash(*m2));
			delete m1;
			delete m2;

			// Test 4: Two matrices with all different values
			m1 = create_matrix(3, 3, 2);
			m2 = create_matrix(3, 3, 2);
			set_all(*m1, 1);
			set_all(*m2, 0);
			Assert::AreNotEqual(matrix_hash(*m1), matrix_hash(*m2));
			delete m1;
			delete m2;
		}
	};
}