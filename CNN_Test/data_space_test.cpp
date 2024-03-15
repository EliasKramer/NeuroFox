#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/data_space.hpp"
#include"test_util.hpp";
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(data_space_test)
	{
	public:
		TEST_METHOD(onlydata_data_space_constructor_test)
		{
			matrix data_format(vector3(2, 2, 1));

			std::vector<matrix> data;

			data_format.set_all(1.0f);
			data.push_back(data_format);
			data_format.set_all(2.0f);
			data.push_back(data_format);

			data_space ds(data_format.get_format(), data);

			Assert::AreEqual((size_t)2, ds.get_item_count());
		}
		TEST_METHOD(data_space_constructor_test)
		{
			matrix data_format(vector3(2, 2, 3));
			matrix label_format(vector3(1, 1, 1));

			std::vector<matrix> data;
			std::vector<matrix> label;

			data_format.set_all(1.0f);
			data.push_back(data_format);

			label_format.set_all(1.5f);
			label.push_back(label_format);

			data_space ds(
				data_format.get_format(),
				label_format.get_format(),
				data,
				label);


			Assert::AreEqual((size_t)1, ds.get_item_count());
		}
		TEST_METHOD(get_only_data)
		{
			matrix data_format(vector3(2, 2, 1));

			std::vector<matrix> data;

			data_format.set_all(1.0f);
			data.push_back(data_format);
			data_format.set_all(2.0f);
			data.push_back(data_format);

			data_space ds(data_format.get_format(), data);

			matrix m1 = matrix(vector3(2, 2, 1));
			m1.set_all(1.0f);
			matrix m2 = matrix(vector3(2, 2, 1));
			m2.set_all(2.0f);


			matrix m(vector3(2, 2, 1));
			ds.observe_data_at_idx(m, 0);
			Assert::IsTrue(matrix::are_equal(m1, m));

			ds.observe_data_at_idx(m, 1);
			Assert::IsTrue(matrix::are_equal(m2, m));
		}
		TEST_METHOD(get_data_and_label_test)
		{
			matrix tmp_data(vector3(2, 2, 3));
			matrix tmp_label(vector3(1, 2, 1));

			std::vector<matrix> data_vector;
			std::vector<matrix> label_vector;

			//set data vector at idx 0:
			/*
				0.900000 1.000000 
				1.000000 1.000000 
				
				1.000000 1.000000 
				1.000000 1.000000 
				
				1.000000 1.000000 
				1.000000 1.000000
			*/
			tmp_data.set_all(1.0f);
			tmp_data.set_at_flat_host(0, 0.9f);
			data_vector.push_back(tmp_data);

			// set label vector at idx 0:
			/*
				0.400000
				1.500000
			*/
			tmp_label.set_all(1.5f);
			tmp_label.set_at_flat_host(0, 0.4f);
			label_vector.push_back(tmp_label);
			
			//set data vector at idx 1:
			/*
				4.900000 5.000000
				5.000000 5.000000

				5.000000 5.000000
				5.000000 5.000000

				5.000000 5.000000
				5.000000 5.000000
			*/
			tmp_data.set_all(5.0f);
			tmp_data.set_at_flat_host(0, 4.9f);
			data_vector.push_back(tmp_data);
			
			// set label vector at idx 1:
			/*
				5.400000
				5.500000
			*/
			tmp_label.set_all(5.5f);
			tmp_label.set_at_flat_host(0, 5.4f);
			label_vector.push_back(tmp_label);

			data_space ds(
				tmp_data.get_format(),
				tmp_label.get_format(),
				data_vector,
				label_vector);

			matrix expected_data(vector3(2, 2, 3));
			matrix expected_label(vector3(1, 2, 1));

			matrix m(vector3(2, 2, 3));
			ds.observe_data_at_idx(m, 0);
			matrix l(vector3(1, 2, 1));
			ds.observe_label_at_idx(l, 0);

			//expected
			expected_data.set_all(1.0f);
			expected_data.set_at_flat_host(0, 0.9f);
			expected_label.set_all(1.5f);
			expected_label.set_at_flat_host(0, 0.4f);


			Assert::IsTrue(matrix::are_equal(m, expected_data), 
				string_to_wstring("\ngot data: \n" + m.get_string() + "\nexpected: \n" + expected_data.get_string()).c_str());
			Assert::IsTrue(matrix::are_equal(l, expected_label), 
				string_to_wstring("\ngot label: \n" + l.get_string() + "\nexpected: \n" + expected_label.get_string()).c_str());

			ds.observe_data_at_idx(m, 1);
			ds.observe_label_at_idx(l, 1);
			expected_data.set_all(5.0f);
			expected_data.set_at_flat_host(0, 4.9f);
			expected_label.set_all(5.5f);
			expected_label.set_at_flat_host(0, 5.4f);
			Assert::IsTrue(matrix::are_equal(m, expected_data));
			Assert::IsTrue(matrix::are_equal(l, expected_label));
		}
		TEST_METHOD(observing_test)
		{
			vector3 data_format(3, 3, 1);
			vector3 label_format(2, 1, 1);

			//setup
			data_space original(
				data_format,
				label_format,
				std::vector<matrix> {
				matrix(data_format,
					std::vector<float> {
					1, 2, 3,
						4, 5, 6,
						7, 8, 9}),
					matrix(data_format,
						std::vector<float> {
					12, 22, 32,
						42, 52, 62,
						72, 82, 92}),
					matrix(data_format,
						std::vector<float> {
					123, 223, 323,
						423, 523, 623,
						723, 823, 923}),
			},
				std::vector<matrix> {
				matrix(label_format, std::vector<float> {
					1, 2
				}),
					matrix(label_format, std::vector<float> {
					21, 22
				}),
					matrix(label_format, std::vector<float> {
					312, 322
				}),
			}
			);

			//execution
			data_space observed(original, 1, 2);

			//test 1
			matrix observed_data(data_format);
			matrix observed_label(label_format);

			observed.observe_data_at_idx(observed_data, 0);
			observed.observe_label_at_idx(observed_label, 0);

			matrix expected_observe(data_format,
				std::vector<float> {
				12, 22, 32,
					42, 52, 62,
					72, 82, 92});

			matrix expected_label(label_format, std::vector<float> {
				21, 22
			});

			Assert::IsTrue(matrix::are_equal(observed_data, expected_observe));
			Assert::IsTrue(matrix::are_equal(observed_label, expected_label));

			//test 2
			observed_data = matrix(data_format);
			observed_label = matrix(label_format);

			observed.observe_data_at_idx(observed_data, 1);
			observed.observe_label_at_idx(observed_label, 1);

			expected_observe = matrix(data_format,
				std::vector<float> {
				123, 223, 323,
					423, 523, 623,
					723, 823, 923}),

				expected_label = matrix(label_format, std::vector<float> {
				312, 322
			});

			Assert::IsTrue(matrix::are_equal(observed_data, expected_observe));
			Assert::IsTrue(matrix::are_equal(observed_label, expected_label));
		}
	};
}