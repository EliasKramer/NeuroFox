#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/matrix.hpp"
#include "test_util.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(matrix_test)
	{
	public:

		TEST_METHOD(creating_matrix)
		{
			matrix m(vector3(2, 3, 4));
			Assert::AreEqual((size_t)2, m.get_width());
			Assert::AreEqual((size_t)3, m.get_height());
			Assert::AreEqual((size_t)4, m.get_depth());
			Assert::AreEqual((size_t)24, m.item_count());
		}
		TEST_METHOD(constructor_from_vector_test)
		{
			matrix m(vector3(2, 3, 4));
			for (float i = 0; i < 24; i++)
			{
				m.set_at_flat(i, i);
			}
			std::vector<float> expected_values(24);
			for (float i = 0; i < 24; i++)
			{
				expected_values[i] = i;
			}
			matrix mx(vector3(2, 3, 4), expected_values);
			Assert::IsTrue(matrix::are_equal(m, mx));
		}
		TEST_METHOD(constructor_from_vector_test_2)
		{
			std::vector<float> initial_data = {
				11, 22,
				33, 44,

				55, 66,
				77, 88
			};

			matrix initial_matrix(vector3(2, 2, 2), initial_data);

			Assert::AreEqual(11.0f, initial_matrix.get_at_host(vector3(0, 0, 0)));
			Assert::AreEqual(22.0f, initial_matrix.get_at_host(vector3(1, 0, 0)));
			Assert::AreEqual(33.0f, initial_matrix.get_at_host(vector3(0, 1, 0)));
			Assert::AreEqual(44.0f, initial_matrix.get_at_host(vector3(1, 1, 0)));
			
			Assert::AreEqual(55.0f, initial_matrix.get_at_host(vector3(0, 0, 1)));
			Assert::AreEqual(66.0f, initial_matrix.get_at_host(vector3(1, 0, 1)));
			Assert::AreEqual(77.0f, initial_matrix.get_at_host(vector3(0, 1, 1)));
			Assert::AreEqual(88.0f, initial_matrix.get_at_host(vector3(1, 1, 1)));
		}
		TEST_METHOD(setting_getting)
		{
			matrix m(vector3(2, 3, 4));
			m.set_at(vector3(0, 0, 0), 1);
			m.set_at(vector3(1, 0, 0), 2);
			m.set_at(vector3(0, 1, 0), 3);
			m.set_at(vector3(1, 1, 0), 4);
			m.set_at(vector3(0, 2, 0), 5);
			m.set_at(vector3(1, 2, 0), 6);
			m.set_at(vector3(0, 0, 1), 7);
			m.set_at(vector3(1, 0, 1), 8);
			m.set_at(vector3(0, 1, 1), 9);
			m.set_at(vector3(1, 1, 1), 10);
			m.set_at(vector3(0, 2, 1), 11);
			m.set_at(vector3(1, 2, 1), 12);
			m.set_at(vector3(0, 0, 2), 13);
			m.set_at(vector3(1, 0, 2), 14);
			m.set_at(vector3(0, 1, 2), 15);
			m.set_at(vector3(1, 1, 2), 16);
			m.set_at(vector3(0, 2, 2), 17);
			m.set_at(vector3(1, 2, 2), 18);
			m.set_at(vector3(0, 0, 3), 19);
			m.set_at(vector3(1, 0, 3), 20);
			m.set_at(vector3(0, 1, 3), 21);
			m.set_at(vector3(1, 1, 3), 22);
			m.set_at(vector3(0, 2, 3), 23);
			m.set_at(vector3(1, 2, 3), 24);
			Assert::AreEqual(2, (int)m.get_at_host(vector3(1, 0, 0)));
			Assert::AreEqual(1, (int)m.get_at_host(vector3(0, 0, 0)));
			Assert::AreEqual(3, (int)m.get_at_host(vector3(0, 1, 0)));
			Assert::AreEqual(4, (int)m.get_at_host(vector3(1, 1, 0)));
			Assert::AreEqual(5, (int)m.get_at_host(vector3(0, 2, 0)));
			Assert::AreEqual(6, (int)m.get_at_host(vector3(1, 2, 0)));
			Assert::AreEqual(7, (int)m.get_at_host(vector3(0, 0, 1)));
			Assert::AreEqual(8, (int)m.get_at_host(vector3(1, 0, 1)));
			Assert::AreEqual(9, (int)m.get_at_host(vector3(0, 1, 1)));
			Assert::AreEqual(10, (int)m.get_at_host(vector3(1, 1, 1)));
			Assert::AreEqual(11, (int)m.get_at_host(vector3(0, 2, 1)));
			Assert::AreEqual(12, (int)m.get_at_host(vector3(1, 2, 1)));
			Assert::AreEqual(13, (int)m.get_at_host(vector3(0, 0, 2)));
			Assert::AreEqual(14, (int)m.get_at_host(vector3(1, 0, 2)));
			Assert::AreEqual(15, (int)m.get_at_host(vector3(0, 1, 2)));
			Assert::AreEqual(16, (int)m.get_at_host(vector3(1, 1, 2)));
			Assert::AreEqual(17, (int)m.get_at_host(vector3(0, 2, 2)));
			Assert::AreEqual(18, (int)m.get_at_host(vector3(1, 2, 2)));
			Assert::AreEqual(19, (int)m.get_at_host(vector3(0, 0, 3)));
			Assert::AreEqual(20, (int)m.get_at_host(vector3(1, 0, 3)));
			Assert::AreEqual(21, (int)m.get_at_host(vector3(0, 1, 3)));
			Assert::AreEqual(22, (int)m.get_at_host(vector3(1, 1, 3)));
			Assert::AreEqual(23, (int)m.get_at_host(vector3(0, 2, 3)));
			Assert::AreEqual(24, (int)m.get_at_host(vector3(1, 2, 3)));
		}
		TEST_METHOD(equal_test)
		{
			matrix original_matrix(vector3(2, 5, 3));
			matrix equal_matrix(vector3(2, 5, 3));
			matrix one_different(vector3(2, 5, 3));
			matrix size_different(vector3(2, 5, 2));

			original_matrix.set_all(1);
			equal_matrix.set_all(1);
			one_different.set_all(1);
			size_different.set_all(1);

			one_different.set_at(vector3(0, 0, 0), 2);

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
			matrix m1(vector3(2, 4, 1));
			matrix m2(vector3(3, 2, 1));

			m1.set_all(3);
			m2.set_all(2);

			matrix m3(vector3(3, 4, 1));
			matrix::dot_product(m1, m2, m3);

			for (int i = 0; i < m3.item_count(); i++)
			{
				Assert::AreEqual(12.0f, m3.get_at_flat_host(i));
			}
		}
		TEST_METHOD(dot_test_3D)
		{
			matrix m1(vector3(2, 4, 2));
			matrix m2(vector3(3, 2, 2));

			m1.set_all(3);
			m2.set_all(2);

			matrix m3(vector3(3, 4, 2));
			matrix::dot_product(m1, m2, m3);

			for (int i = 0; i < m3.item_count(); i++)
			{
				Assert::AreEqual(12.0f, m3.get_at_flat_host(i));
			}
		}
		TEST_METHOD(valid_cross_correlation_test)
		{
			/* input matrix
				+ - + - +
				| 1 | 3 |
				+ - + - +
				| 2 | 4 |
				+ - + - +
			*/
			matrix kernel(vector3(2, 2, 1));

			kernel.set_at(vector3(0, 0), 1);
			kernel.set_at(vector3(0, 1), 2);
			kernel.set_at(vector3(1, 0), 3);
			kernel.set_at(vector3(1, 1), 4);

			matrix input(vector3(3, 3, 1));

			/* input matrix
				+ - + - + - +
				| 1 | 4 | 7 |
				+ - + - + - +
				| 2 | 5 | 8 |
				+ - + - + - +
				| 3 | 6 | 9 |
				+ - + - + - +
			*/

			input.set_at(vector3(0, 0), 1);
			input.set_at(vector3(0, 1), 2);
			input.set_at(vector3(0, 2), 3);
			input.set_at(vector3(1, 0), 4);
			input.set_at(vector3(1, 1), 5);
			input.set_at(vector3(1, 2), 6);
			input.set_at(vector3(2, 0), 7);
			input.set_at(vector3(2, 1), 8);
			input.set_at(vector3(2, 2), 9);

			matrix output(vector3(2, 2, 1));
			output.set_all(0);

			//1*1 + 4*3 + 2*2 + 5*4 = 37 (0,0)
			//1*4 + 3*7 + 2*5 + 4*8 = 67 (1,0)
			//1*2 + 3*5 + 2*3 + 4*6 = 47 (0,1)
			//1*5 + 3*8 + 2*6 + 4*9 = 77 (1,1)

			//expected output 
			/*
				+ -- + -- +
				| 37 | 67 |
				+ -- + -- +
				| 47 | 77 |
				+ -- + -- +
			*/
			std::vector <matrix> kernels;
			kernels.push_back(kernel);

			matrix::valid_cross_correlation(input, kernels, output, 1);

			Assert::AreEqual(37, (int)output.get_at_host(vector3(0, 0)));
			Assert::AreEqual(67, (int)output.get_at_host(vector3(1, 0)));
			Assert::AreEqual(47, (int)output.get_at_host(vector3(0, 1)));
			Assert::AreEqual(77, (int)output.get_at_host(vector3(1, 1)));
		}
		TEST_METHOD(test_copy)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(1);
			matrix m2 = matrix(m);
			Assert::IsTrue(matrix::are_equal(m, m2));

			m2.set_at(vector3(0, 0, 0), 2);
			Assert::AreNotEqual(2.0f, m.get_at_host(vector3(0, 0, 0)));

			m.set_at(vector3(0, 0, 1), 3);
			Assert::AreNotEqual(3.0f, m2.get_at_host(vector3(0, 0, 1)));
		}
	};
}