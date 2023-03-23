#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/neural_network.hpp"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(PoolingTest)
	{
	public:
		
		TEST_METHOD(PoolingTest1)
		{
			pooling_layer* pool_layer = create_pooling_layer(2, 2, max);
		}
	};
}
