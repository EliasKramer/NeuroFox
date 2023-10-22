#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/matrix.hpp"
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
				m.set_at_flat_host(i, i);
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
			m.set_at_host(vector3(0, 0, 0), 1);
			m.set_at_host(vector3(1, 0, 0), 2);
			m.set_at_host(vector3(0, 1, 0), 3);
			m.set_at_host(vector3(1, 1, 0), 4);
			m.set_at_host(vector3(0, 2, 0), 5);
			m.set_at_host(vector3(1, 2, 0), 6);
			m.set_at_host(vector3(0, 0, 1), 7);
			m.set_at_host(vector3(1, 0, 1), 8);
			m.set_at_host(vector3(0, 1, 1), 9);
			m.set_at_host(vector3(1, 1, 1), 10);
			m.set_at_host(vector3(0, 2, 1), 11);
			m.set_at_host(vector3(1, 2, 1), 12);
			m.set_at_host(vector3(0, 0, 2), 13);
			m.set_at_host(vector3(1, 0, 2), 14);
			m.set_at_host(vector3(0, 1, 2), 15);
			m.set_at_host(vector3(1, 1, 2), 16);
			m.set_at_host(vector3(0, 2, 2), 17);
			m.set_at_host(vector3(1, 2, 2), 18);
			m.set_at_host(vector3(0, 0, 3), 19);
			m.set_at_host(vector3(1, 0, 3), 20);
			m.set_at_host(vector3(0, 1, 3), 21);
			m.set_at_host(vector3(1, 1, 3), 22);
			m.set_at_host(vector3(0, 2, 3), 23);
			m.set_at_host(vector3(1, 2, 3), 24);
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

			one_different.set_at_host(vector3(0, 0, 0), 2);

			Assert::IsTrue(matrix::are_equal(original_matrix, original_matrix));

			Assert::IsTrue(matrix::are_equal(original_matrix, equal_matrix));
			Assert::IsTrue(matrix::are_equal(equal_matrix, original_matrix));

			Assert::IsFalse(matrix::are_equal(original_matrix, one_different));
			Assert::IsFalse(matrix::are_equal(one_different, original_matrix));

			Assert::IsFalse(matrix::are_equal(original_matrix, size_different));
			Assert::IsFalse(matrix::are_equal(size_different, original_matrix));
		}
		TEST_METHOD(dot_product_flat)
		{
			matrix input(vector3(2, 2, 2), std::vector<float> {
				0, 
				1,
				2,
				3,

				4, 
				5,
				6, 
				7
			});

			matrix result(vector3(1, 2, 3));
			result.set_all(0);

			matrix weights(vector3(input.item_count(), result.item_count(), 1), std::vector<float> {
				0, 1, 2, 3, 4, 5, 6, 7,
				8, 9, 0, 1, 2, 3, 4, 5,
				6, 7, 8, 9, 0, 1, 2, 3,
				4, 5, 6, 7, 8, 9, 0, 1,
				2, 3, 4, 5, 6, 7, 8, 9,
				0, 1, 2, 3, 4, 5, 6, 7
			});

			matrix::dot_product_flat(weights, input, result);
			/*
										0,
										1,
										2,
										3,
										4,
										5,
										6,
										7
				0, 1, 2, 3, 4, 5, 6, 7| 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 = 140
				8, 9, 0, 1, 2, 3, 4, 5| 8*0 + 9*1 + 0*2 + 1*3 + 2*4 + 3*5 + 4*6 + 5*7 = 94
				6, 7, 8, 9, 0, 1, 2, 3| 6*0 + 7*1 + 8*2 + 9*3 + 0*4 + 1*5 + 2*6 + 3*7 = 88
				4, 5, 6, 7, 8, 9, 0, 1| 4*0 + 5*1 + 6*2 + 7*3 + 8*4 + 9*5 + 0*6 + 1*7 = 122
				2, 3, 4, 5, 6, 7, 8, 9| 2*0 + 3*1 + 4*2 + 5*3 + 6*4 + 7*5 + 8*6 + 9*7 = 196
				0, 1, 2, 3, 4, 5, 6, 7| 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 = 140
			*/

			Assert::AreEqual(140.0f, result.get_at_flat_host(0));
			Assert::AreEqual(94.0f, result.get_at_flat_host(1));
			Assert::AreEqual(88.0f, result.get_at_flat_host(2));
			Assert::AreEqual(122.0f, result.get_at_flat_host(3));
			Assert::AreEqual(196.0f, result.get_at_flat_host(4));
			Assert::AreEqual(140.0f, result.get_at_flat_host(5));
		}
		TEST_METHOD(dot_product_flat_gpu)
		{
			matrix input(vector3(2, 2, 2), std::vector<float> {
				0, 1,
					2, 3,

					4, 5,
					6, 7
			});

			matrix result(vector3(1, 2, 3));
			result.set_all(0);

			matrix weights(vector3(input.item_count(), result.item_count(), 1), std::vector<float> {
				0, 1, 2, 3, 4, 5, 6, 7,
					8, 9, 0, 1, 2, 3, 4, 5,
					6, 7, 8, 9, 0, 1, 2, 3,
					4, 5, 6, 7, 8, 9, 0, 1,
					2, 3, 4, 5, 6, 7, 8, 9,
					0, 1, 2, 3, 4, 5, 6, 7
			});

			input.enable_gpu_mode();
			result.enable_gpu_mode();
			weights.enable_gpu_mode();

			matrix::dot_product_flat(weights, input, result);
			/*
										0,
										1,
										2,
										3,
										4,
										5,
										6,
										7
				0, 1, 2, 3, 4, 5, 6, 7| 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 = 140
				8, 9, 0, 1, 2, 3, 4, 5| 8*0 + 9*1 + 0*2 + 1*3 + 2*4 + 3*5 + 4*6 + 5*7 = 94
				6, 7, 8, 9, 0, 1, 2, 3| 6*0 + 7*1 + 8*2 + 9*3 + 0*4 + 1*5 + 2*6 + 3*7 = 88
				4, 5, 6, 7, 8, 9, 0, 1| 4*0 + 5*1 + 6*2 + 7*3 + 8*4 + 9*5 + 0*6 + 1*7 = 122
				2, 3, 4, 5, 6, 7, 8, 9| 2*0 + 3*1 + 4*2 + 5*3 + 6*4 + 7*5 + 8*6 + 9*7 = 196
				0, 1, 2, 3, 4, 5, 6, 7| 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 = 140
			*/
			result.sync_device_and_host();

			Assert::AreEqual(140.0f, result.get_at_flat_host(0));
			Assert::AreEqual(94.0f, result.get_at_flat_host(1));
			Assert::AreEqual(88.0f, result.get_at_flat_host(2));
			Assert::AreEqual(122.0f, result.get_at_flat_host(3));
			Assert::AreEqual(196.0f, result.get_at_flat_host(4));
			Assert::AreEqual(140.0f, result.get_at_flat_host(5));
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

			kernel.set_at_host(vector3(0, 0), 1);
			kernel.set_at_host(vector3(0, 1), 2);
			kernel.set_at_host(vector3(1, 0), 3);
			kernel.set_at_host(vector3(1, 1), 4);

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

			input.set_at_host(vector3(0, 0), 1);
			input.set_at_host(vector3(0, 1), 2);
			input.set_at_host(vector3(0, 2), 3);
			input.set_at_host(vector3(1, 0), 4);
			input.set_at_host(vector3(1, 1), 5);
			input.set_at_host(vector3(1, 2), 6);
			input.set_at_host(vector3(2, 0), 7);
			input.set_at_host(vector3(2, 1), 8);
			input.set_at_host(vector3(2, 2), 9);

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

			matrix::cross_correlation(input, kernels, output, 1);

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

			m2.set_at_host(vector3(0, 0, 0), 2);
			Assert::AreNotEqual(2.0f, m.get_at_host(vector3(0, 0, 0)));

			m.set_at_host(vector3(0, 0, 1), 3);
			Assert::AreNotEqual(3.0f, m2.get_at_host(vector3(0, 0, 1)));
		}
		TEST_METHOD(subtract)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(5);
			matrix m2 = matrix(vector3(2, 3, 5));
			m2.set_all(2);

			matrix result(vector3(2, 3, 5));

			matrix::subtract(m, m2, result);

			matrix expected(vector3(2, 3, 5));
			expected.set_all(3);

			Assert::IsTrue(matrix::are_equal(expected, result));
		}
		TEST_METHOD(subtract_negative_result)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(2);
			matrix m2 = matrix(vector3(2, 3, 5));
			m2.set_all(5);

			matrix result(vector3(2, 3, 5));

			matrix::subtract(m, m2, result);

			matrix expected(vector3(2, 3, 5));
			expected.set_all(-3);

			Assert::IsTrue(matrix::are_equal(expected, result));
		}
		TEST_METHOD(subtract_gpu)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(5);
			m.enable_gpu_mode();
			matrix m2 = matrix(vector3(2, 3, 5));
			m2.set_all(2);
			m2.enable_gpu_mode();

			matrix result(vector3(2, 3, 5));
			result.enable_gpu_mode();

			matrix::subtract(m, m2, result);
			result.sync_device_and_host();

			matrix expected(vector3(2, 3, 5));
			expected.set_all(3);


			Assert::IsTrue(matrix::are_equal(expected, result));
		}
		TEST_METHOD(subtract_negative_result_gpu)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(2);
			m.enable_gpu_mode();
			matrix m2 = matrix(vector3(2, 3, 5));
			m2.set_all(5);
			m2.enable_gpu_mode();

			matrix result(vector3(2, 3, 5));
			result.enable_gpu_mode();

			matrix::subtract(m, m2, result);
			result.sync_device_and_host();

			matrix expected(vector3(2, 3, 5));
			expected.set_all(-3);

			Assert::IsTrue(matrix::are_equal(expected, result));
		}
		TEST_METHOD(activation_sigmoid)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(1);
			m.set_at_flat_host(0, 0);

			m.apply_activation_function(e_activation_t::sigmoid_fn);

			matrix expected(vector3(2, 3, 5));
			expected.set_all(sigmoid(1));
			expected.set_at_flat_host(0, sigmoid(0));

			Assert::IsTrue(matrix::are_equal(expected, m));
		}
		TEST_METHOD(activation_relu)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(-1);
			m.set_at_flat_host(0, 5);

			m.apply_activation_function(e_activation_t::relu_fn);

			matrix expected(vector3(2, 3, 5));
			expected.set_all(relu(-1));
			expected.set_at_flat_host(0, relu(5));

			Assert::IsTrue(matrix::are_equal(expected, m));
		}
		TEST_METHOD(activation_sigmoid_gpu_test)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(1);
			m.set_at_flat_host(0, 0);
			m.enable_gpu_mode();

			m.apply_activation_function(e_activation_t::sigmoid_fn);
			m.sync_device_and_host();

			matrix expected(vector3(2, 3, 5));
			expected.set_all(sigmoid(1));
			expected.set_at_flat_host(0, sigmoid(0));

			Assert::IsTrue(matrix::are_equal(expected, m));
		}
		TEST_METHOD(activation_relu_gpu_test)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(-1);
			m.set_at_flat_host(0, 5);
			m.enable_gpu_mode();

			m.apply_activation_function(e_activation_t::relu_fn);
			m.sync_device_and_host();

			matrix expected(vector3(2, 3, 5));
			expected.set_all(relu(-1));
			expected.set_at_flat_host(0, relu(5));

			Assert::IsTrue(matrix::are_equal(expected, m));
		}
		TEST_METHOD(not_all_items_zero_host_test)
		{
			matrix m = matrix(vector3(2, 3, 5));
			m.set_all(0);

			Assert::IsFalse(m.contains_non_zero_items());

			m.set_at_flat_host(0, 1);
			Assert::IsTrue(m.contains_non_zero_items());
		}
		TEST_METHOD(not_all_items_zero_device_test)
		{
			matrix m = matrix(vector3(100, 3, 5));
			m.set_all(0);
			m.enable_gpu_mode();

			Assert::IsFalse(m.contains_non_zero_items());

			m.set_at_flat_host(m.item_count() - 1, 1);
			m.sync_device_and_host();

			Assert::IsTrue(m.contains_non_zero_items());
		}
		void test_add_flat_helper(int n)
		{
			std::vector <float> v1(n);
			std::vector <float> v2(n);
			std::vector <float> expected(n);
			std::vector <float> result(n);

			for (int i = 0; i < n; i++)
			{
				v1[i] = i;
				v2[i] = i;
				expected[i] = i + i;
				result[i] = 0;
			}

			matrix m1(vector3(1, n, 1), v1);
			matrix m2(vector3(1, n, 1), v2);
			matrix m3(vector3(1, n, 1), result);

			matrix::add_flat(m1, m2, m3);
			matrix expected_m(vector3(1, n, 1), expected);

			Assert::IsTrue(matrix::are_equal(expected_m, m3));
		}
		TEST_METHOD(test_add_flat)
		{
			test_add_flat_helper(1);
			test_add_flat_helper(2);
			test_add_flat_helper(3);
			test_add_flat_helper(4);
			test_add_flat_helper(5);
			test_add_flat_helper(6);
			test_add_flat_helper(7);
			test_add_flat_helper(8);
			test_add_flat_helper(9);
			test_add_flat_helper(100);
			test_add_flat_helper(1000);
			test_add_flat_helper(1024);
		}
		void test_substract_flat_helper(int n)
		{
			std::vector <float> v1(n);
			std::vector <float> v2(n);
			std::vector <float> expected(n);
			std::vector <float> result(n);

			for (int i = 0; i < n; i++)
			{
				v1[i] = i * 1.3;
				v2[i] = i * 0.6;
				expected[i] = v1[i] - v2[i];
				result[i] = 0;
			}

			matrix m1(vector3(1, n, 1), v1);
			matrix m2(vector3(1, n, 1), v2);
			matrix m3(vector3(1, n, 1), result);

			matrix::subtract_flat(m1, m2, m3);
			matrix expected_m(vector3(1, n, 1), expected);

			Assert::IsTrue(matrix::are_equal(expected_m, m3));
		}
		TEST_METHOD(test_sub_flat)
		{
			test_add_flat_helper(1);
			test_add_flat_helper(2);
			test_add_flat_helper(3);
			test_add_flat_helper(4);
			test_add_flat_helper(5);
			test_add_flat_helper(6);
			test_add_flat_helper(7);
			test_add_flat_helper(8);
			test_add_flat_helper(9);
			test_add_flat_helper(100);
			test_add_flat_helper(1000);
			test_add_flat_helper(1024);
		}
		void scalar_product_helper(int n, float scalar)
		{
			std::vector <float> v1(n);
			std::vector <float> expected(n);
			std::vector <float> result(n);

			for (int i = 0; i < n; i++)
			{
				v1[i] = i * 1.25;
				expected[i] = v1[i] * scalar;
				result[i] = 0;
			}

			matrix m1(vector3(1, n, 1), v1);
			matrix expected_m(vector3(1, n, 1), expected);

			m1.scalar_multiplication(scalar);

			Assert::IsTrue(matrix::are_equal(expected_m, m1, 0.1f));
		}
		TEST_METHOD(sclar_product_test)
		{
			scalar_product_helper(2, 1.3f);
			scalar_product_helper(3, 2);
			scalar_product_helper(4, 13.3094f);
			scalar_product_helper(1, 4 * 3.141592653);
			scalar_product_helper(5, 63.2);
			scalar_product_helper(6, 1);
			scalar_product_helper(7, 8.854);
			scalar_product_helper(8, 90);
			scalar_product_helper(9, 931);
			scalar_product_helper(10, 9.10938);
			scalar_product_helper(100, 9.84);
			scalar_product_helper(1000, 3.14159);
			scalar_product_helper(1024, 8.314);
		}
	};
}