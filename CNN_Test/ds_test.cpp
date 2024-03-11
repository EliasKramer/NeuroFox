#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/code/data_space.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(ds_test)
	{
	public:

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
						1,2,3,
						4,5,6,
						7,8,9}),
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
					1,2
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
