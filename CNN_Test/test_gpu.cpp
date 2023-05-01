#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/gpu_math.cuh"
#include "test_util.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(cuda_test)
	{
	public:
		std::vector<float> get_gpu_values(float* gpu_ptr, int n)
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
			gpu_mem.set_all(0xC00FFEE);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.gpu_data_ptr(), gpu_mem.count());
			std::vector<float> expected_values(gpu_mem.count(), 0xC00FFEE);
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
			m.set_all(0xDEADBEEF);

			gpu_memory<float> gpu_mem(m);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.gpu_data_ptr(), m.flat_readonly().size());
			std::vector<float> expected_values(m.flat_readonly().size(), 0xDEADBEEF);
			Assert::IsTrue(float_vectors_equal(expected_values, gpu_values));
		}
		TEST_METHOD(add_matrix_gpu_test)
		{
			int n = 1000000;
			gpu_memory<float> gpu_mem_a(n);
			gpu_memory<float> gpu_mem_b(n);
			gpu_memory<float> gpu_mem_result(n);

			gpu_mem_a.set_all(1);
			gpu_mem_b.set_all(2);
			gpu_mem_result.set_all(0);

			cudaError_t cuda_error = gpu_add(gpu_mem_a, gpu_mem_b, gpu_mem_result);
			Assert::AreEqual((int)cudaSuccess, (int)cuda_error);

			std::vector<float> result = *gpu_mem_result.to_cpu().get();
			std::vector<float> expected(n, 3);
			Assert::IsTrue(float_vectors_equal(expected, result));
		}
	};
}