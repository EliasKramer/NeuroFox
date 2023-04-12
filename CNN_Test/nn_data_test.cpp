#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/nn_data.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(data_test)
	{
	public:

		TEST_METHOD(constructor_setting_and_getting)
		{
			matrix data = get_matrix(1, 1, 1);
			matrix label = get_matrix(1, 1, 1);
			nn_data d(data, label);

			// test if the data is set correctly
			Assert::IsTrue(matrix_equal_format(d.get_data(), data));
			Assert::IsTrue(matrix_equal_format(d.get_label(), label));

			// test if the label can be set through setter
			matrix new_data = get_matrix(1, 2, 1);
			matrix new_label = get_matrix(1, 2, 1);
			d.set_data(new_data);
			d.set_label(new_label);
			Assert::IsTrue(matrix_equal_format(d.get_data(), new_data));
			Assert::IsTrue(matrix_equal_format(d.get_label(), new_label));
		}
		TEST_METHOD(constructor_copies_not_references)
		{
			matrix data = get_matrix(1, 1, 1);
			matrix label = get_matrix(1, 1, 1);

			//set the data before the constructor to 1
			data.data[0] = 1.0f;
			label.data[0] = 1.0f;

			//construct the data
			nn_data d(data, label);

			//set the data that was used to something else
			data.data[0] = 2.0f;
			label.data[0] = 2.0f;

			//the nn_data shoud stay the same, because it got copied
			Assert::AreEqual(d.get_data().data[0], 1.0f);
			Assert::AreEqual(d.get_label().data[0], 1.0f);
			Assert::AreEqual(data.data[0], 2.0f);
			Assert::AreEqual(label.data[0], 2.0f);
		}
	};
}
