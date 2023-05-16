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
			matrix data(1, 1, 1);
			matrix label(1, 1, 1);
			nn_data d(data, label);
			// test if the data is set correctly
			Assert::IsTrue(matrix::equal_format(d.get_data(), data));
			Assert::IsTrue(matrix::equal_format(d.get_label(), label));

			// test if the label can be set through setter
			matrix new_data(1, 2, 1);
			matrix new_label(1, 2, 1);
			d.set_data(new_data);
			d.set_label(new_label);
			Assert::IsTrue(matrix::equal_format(d.get_data(), new_data));
			Assert::IsTrue(matrix::equal_format(d.get_label(), new_label));
		}
		TEST_METHOD(constructor_copies_not_references)
		{
			matrix data(1, 1, 1);
			matrix label(1, 1, 1);

			//set the data before the constructor to 1
			data.set_at_flat(0, 1.0f);
			label.set_at_flat(0, 1.0f);

			//construct the data
			nn_data d(data, label);

			//set the data that was used to something else
			data.set_at_flat(0, 2.0f);
			label.set_at_flat(0, 2.0f);

			//the nn_data shoud stay the same, because it got copied
			Assert::AreEqual(1.0f, d.get_data().get_at_flat(0));
			Assert::AreEqual(1.0f, d.get_label().get_at_flat(0));
			Assert::AreEqual(2.0f, data.get_at_flat(0));
			Assert::AreEqual(2.0f, label.get_at_flat(0));
		}
	};
}
