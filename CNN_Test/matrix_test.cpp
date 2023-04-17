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
			matrix m(2, 3, 4);
			Assert::AreEqual(2, m.get_width());
			Assert::AreEqual(3, m.get_height());
			Assert::AreEqual(4, m.get_depth());
			Assert::AreEqual(24, (int)m.flat_readonly().size());
		}
		TEST_METHOD(setting_getting)
		{
			matrix m(2, 3, 4);
			m.set_at(0, 0, 0, 1);
			m.set_at(1, 0, 0, 2);
			m.set_at(0, 1, 0, 3);
			m.set_at(1, 1, 0, 4);
			m.set_at(0, 2, 0, 5);
			m.set_at(1, 2, 0, 6);
			m.set_at(0, 0, 1, 7);
			m.set_at(1, 0, 1, 8);
			m.set_at(0, 1, 1, 9);
			m.set_at(1, 1, 1, 10);
			m.set_at(0, 2, 1, 11);
			m.set_at(1, 2, 1, 12);
			m.set_at(0, 0, 2, 13);
			m.set_at(1, 0, 2, 14);
			m.set_at(0, 1, 2, 15);
			m.set_at(1, 1, 2, 16);
			m.set_at(0, 2, 2, 17);
			m.set_at(1, 2, 2, 18);
			m.set_at(0, 0, 3, 19);
			m.set_at(1, 0, 3, 20);
			m.set_at(0, 1, 3, 21);
			m.set_at(1, 1, 3, 22);
			m.set_at(0, 2, 3, 23);
			m.set_at(1, 2, 3, 24);
			Assert::AreEqual(2, (int)m.get_at(1, 0, 0));
			Assert::AreEqual(1, (int)m.get_at(0, 0, 0));
			Assert::AreEqual(3, (int)m.get_at(0, 1, 0));
			Assert::AreEqual(4, (int)m.get_at(1, 1, 0));
			Assert::AreEqual(5, (int)m.get_at(0, 2, 0));
			Assert::AreEqual(6, (int)m.get_at(1, 2, 0));
			Assert::AreEqual(7, (int)m.get_at(0, 0, 1));
			Assert::AreEqual(8, (int)m.get_at(1, 0, 1));
			Assert::AreEqual(9, (int)m.get_at(0, 1, 1));
			Assert::AreEqual(10, (int)m.get_at(1, 1, 1));
			Assert::AreEqual(11, (int)m.get_at(0, 2, 1));
			Assert::AreEqual(12, (int)m.get_at(1, 2, 1));
			Assert::AreEqual(13, (int)m.get_at(0, 0, 2));
			Assert::AreEqual(14, (int)m.get_at(1, 0, 2));
			Assert::AreEqual(15, (int)m.get_at(0, 1, 2));
			Assert::AreEqual(16, (int)m.get_at(1, 1, 2));
			Assert::AreEqual(17, (int)m.get_at(0, 2, 2));
			Assert::AreEqual(18, (int)m.get_at(1, 2, 2));
			Assert::AreEqual(19, (int)m.get_at(0, 0, 3));
			Assert::AreEqual(20, (int)m.get_at(1, 0, 3));
			Assert::AreEqual(21, (int)m.get_at(0, 1, 3));
			Assert::AreEqual(22, (int)m.get_at(1, 1, 3));
			Assert::AreEqual(23, (int)m.get_at(0, 2, 3));
			Assert::AreEqual(24, (int)m.get_at(1, 2, 3));
		}
		TEST_METHOD(equal_test)
		{
			matrix original_matrix(2, 5, 3);
			matrix equal_matrix(2, 5, 3);
			matrix one_different(2, 5, 3);
			matrix size_different(2, 5, 2);

			original_matrix.set_all(1);
			equal_matrix.set_all(1);
			one_different.set_all(1);
			size_different.set_all(1);

			one_different.set_at(0, 0, 0, 2);

			Assert::IsTrue(matrix::are_equal(original_matrix, original_matrix));

			Assert::IsTrue(matrix::are_equal(original_matrix, equal_matrix));
			Assert::IsTrue(matrix::are_equal(equal_matrix, original_matrix));

			Assert::IsFalse(matrix::are_equal(original_matrix, one_different));
			Assert::IsFalse(matrix::are_equal(one_different, original_matrix));

			Assert::IsFalse(matrix::are_equal(original_matrix, size_different));
			Assert::IsFalse(matrix::are_equal(size_different, original_matrix));
		}
		TEST_METHOD(dot_test_2D)
		{
			matrix m1(2, 4, 1);
			matrix m2(3, 2, 1);

			m1.set_all(3);
			m2.set_all(2);

			matrix m3(3, 4, 1);
			matrix::dot_product(m1, m2, m3);

			for (int i = 0; i < m3.flat_readonly().size(); i++)
			{
				Assert::AreEqual(12, (int)m3.flat_readonly()[i]);
			}
		}
		TEST_METHOD(dot_test_3D)
		{
			matrix m1(2, 4, 2);
			matrix m2(3, 2, 2);

			m1.set_all(3);
			m2.set_all(2);

			matrix m3(3, 4, 2);
			matrix::dot_product(m1, m2, m3);

			for (int i = 0; i < m3.flat_readonly().size(); i++)
			{
				Assert::AreEqual(12, (int)m3.flat_readonly()[i]);
			}
		}
		TEST_METHOD(hash_test)
		{
			// Test 1: Two identical matrices
			matrix m1(2, 4, 2);
			matrix m2(2, 4, 2);
			m1.set_all(3);
			m2.set_all(3);
			Assert::AreEqual(m1.get_hash(), m2.get_hash());

			// Test 2: Two matrices with one different value
			m1 = matrix(2, 4, 2);
			m2 = matrix(2, 4, 2);
			m1.set_all(3);
			m2.set_all(3);
			m2.set_at(0, 0, 0, 4);
			Assert::AreNotEqual(m1.get_hash(), m2.get_hash());

			// Test 3: Two matrices with different sizes
			m1 = matrix(2, 4, 2);
			m2 = matrix(2, 3, 2);
			m1.set_all(0);
			m2.set_all(0);
			Assert::AreNotEqual(m1.get_hash(), m2.get_hash());

			// Test 4: Two matrices with all different values
			m1 = matrix(3, 3, 2);
			m2 = matrix(3, 3, 2);
			m1.set_all( 1);
			m2.set_all( 0);
			Assert::AreNotEqual(m1.get_hash(), m2.get_hash());
		}
	};
}