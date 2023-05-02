#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/gpu_math.cuh"
#include "test_util.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(cuda_test)
	{
	public:
		std::vector<float> get_gpu_values(float* gpu_ptr, size_t n)
		{
			std::vector<float> result(n);
			cudaError_t cudaStatus = cudaSetDevice(0);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaMemcpy(result.data(), gpu_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			return result;
		}

		TEST_METHOD(gpu_memory_set_test)
		{
			gpu_memory<float> gpu_mem(2);
			gpu_mem.set_all((float)0xC00FFEE);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.gpu_data_ptr(), gpu_mem.item_count());
			std::vector<float> expected_values(gpu_mem.item_count(), (float)0xC00FFEE);
			Assert::IsTrue(float_vectors_equal(expected_values, gpu_values));
		}
		TEST_METHOD(gpu_memory_destructor_test)
		{
			gpu_memory<float> gpu_mem(2);

			float* gpu_ptr = gpu_mem.gpu_data_ptr();
			gpu_mem.~gpu_memory();

			std::vector<float> cpu_data(2);
			cudaError_t cudaStatus =
				cudaMemcpy(
					cpu_data.data(),
					gpu_ptr, 2 * sizeof(float),
					cudaMemcpyDeviceToHost);

			Assert::AreNotEqual((int)cudaSuccess, (int)cudaStatus);
		}
		TEST_METHOD(copy_matrix_to_gpu_test)
		{
			matrix m(2, 3, 4);
			m.set_all((float)0xDEADBEEF);

			gpu_memory<float> gpu_mem(m);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.gpu_data_ptr(), m.flat_readonly().size());
			std::vector<float> expected_values(m.flat_readonly().size(), (float)0xDEADBEEF);
			Assert::IsTrue(float_vectors_equal(expected_values, gpu_values));
		}
		TEST_METHOD(add_gpu_test)
		{
			int n = 1000000;
			gpu_memory<float> gpu_mem_a(n);
			gpu_memory<float> gpu_mem_b(n);
			gpu_memory<float> gpu_mem_result(n);

			gpu_mem_a.set_all(1);
			gpu_mem_b.set_all(2);
			gpu_mem_result.set_all(0);

			gpu_add(gpu_mem_a, gpu_mem_b, gpu_mem_result);

			std::vector<float> result = *gpu_mem_result.to_cpu().get();
			std::vector<float> expected(n, 3);
			Assert::IsTrue(float_vectors_equal(expected, result));
		}
		TEST_METHOD(gpu_activation_relu_test)
		{
			matrix m(1, 3, 1);
			m.set_at(0, 0, 1);
			m.set_at(0, 1, 0);
			m.set_at(0, 2, -3);

			gpu_memory<float> gpu_mem(m);
			GPU_ACTIVATION[relu_fn](gpu_mem);

			std::vector<float> result = *gpu_mem.to_cpu().get();
			std::vector<float> expected = { 1, 0, 0 };
			Assert::IsTrue(float_vectors_equal(expected, result));
		}
		TEST_METHOD(gpu_activation_sigmoid_test)
		{
			matrix m(1, 3, 1);
			m.set_at(0, 0, 1);
			m.set_at(0, 1, 0);
			m.set_at(0, 2, -3);

			gpu_memory<float> gpu_mem(m);
			GPU_ACTIVATION[sigmoid_fn](gpu_mem);

			std::vector<float> result = *gpu_mem.to_cpu().get();
			std::vector<float> expected = {
				ACTIVATION[sigmoid_fn](1),
				ACTIVATION[sigmoid_fn](0),
				ACTIVATION[sigmoid_fn](-3)
			};
			Assert::IsTrue(float_vectors_equal(expected, result));
		}
		TEST_METHOD(dot_gpu_test)
		{

			matrix m_input(1, 2, 1);
			m_input.set_at(0, 0, 5);
			m_input.set_at(0, 1, 6);

			matrix m_weights(2, 3, 1);
			m_weights.set_at(0, 0, 1);
			m_weights.set_at(0, 1, 2);
			m_weights.set_at(0, 2, 3);
			m_weights.set_at(1, 0, 4);
			m_weights.set_at(1, 1, 5);
			m_weights.set_at(1, 2, 6);

			matrix m_activations(1, 3, 1);
			m_activations.set_all(0);

			gpu_memory<float> gpu_input(m_input);
			gpu_memory<float> gpu_weights(m_weights);
			gpu_memory<float> gpu_activations(m_activations);

			//		+-+
			//		|5|
			//		+-+
			//		|6|
			//		+-+
			//+-+-+
			//|1|4|	29 = 1*5 + 4*6
			//+-+-+
			//|2|5|	40 = 2*5 + 5*6
			//+-+-+
			//|3|6| 51 = 3*5 + 6*6
			//+-+-+

			matrix expected_activations(1, 3, 1);
			expected_activations.set_at(0, 0, 29);
			expected_activations.set_at(0, 1, 40);
			expected_activations.set_at(0, 2, 51);

			gpu_dot_product(gpu_weights, gpu_input, gpu_activations);

			std::vector<float> result = *gpu_activations.to_cpu().get();
			std::vector<float> expected = expected_activations.flat_readonly();
			Assert::IsTrue(float_vectors_equal(expected, result));
		}
	};
}