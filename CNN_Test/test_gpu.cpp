#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/gpu_matrix.cuh"
#include "test_util.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(cuda_test)
	{
	public:
		void allocate_gpu_matrices(float** gpu_a, float** gpu_b, float** gpu_result, int n)
		{
			cudaError_t cudaStatus = cudaSetDevice(0);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaMalloc((void**)gpu_a, n * sizeof(float));
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaMalloc((void**)gpu_b, n * sizeof(float));
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaMalloc((void**)gpu_result, n * sizeof(float));
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
		}
		void set_gpu_matrix(float* gpu_ptr, float value, int n)
		{
			cudaError_t cudaStatus = cudaSetDevice(0);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			float* cpu_ptr = new float[n];
			for (int i = 0; i < n; i++)
			{
				cpu_ptr[i] = value;
			}
			cudaStatus = cudaMemcpy(gpu_ptr, cpu_ptr, n * sizeof(float), cudaMemcpyHostToDevice);
			//cudaStatus = cudaMemset(*gpu_ptr, value, n * sizeof(float));
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
		}
		void free_gpu_matrices(float* gpu_a, float* gpu_b, float* gpu_result, int n)
		{
			cudaError_t cudaStatus = cudaSetDevice(0);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaFree(gpu_a);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaFree(gpu_b);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaFree(gpu_result);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
		}
		
		std::vector<float> get_gpu_values(float* gpu_ptr, int n)
		{
			std::vector<float> result(n);
			cudaError_t cudaStatus = cudaSetDevice(0);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			cudaStatus = cudaMemcpy(result.data(), gpu_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
			Assert::AreEqual((int)cudaSuccess, (int)cudaStatus);
			return result;
		}
		TEST_METHOD(add_matrix_gpu_test)
		{
			float* gpu_matrix_a = nullptr;
			float* gpu_matrix_b = nullptr;
			float* gpu_matrix_result = nullptr;

			int n = 10;
			allocate_gpu_matrices(&gpu_matrix_a, &gpu_matrix_b, &gpu_matrix_result, n);
			set_gpu_matrix(gpu_matrix_a, 1, n);
			set_gpu_matrix(gpu_matrix_b, 2, n);
			set_gpu_matrix(gpu_matrix_result, 0, n);

			cudaError_t cuda_error = gpu_add_matrices(gpu_matrix_a, gpu_matrix_b, gpu_matrix_result, (unsigned int)n);
			Assert::AreEqual((int)cudaSuccess, (int)cuda_error);
			
			std::vector<float> result = get_gpu_values(gpu_matrix_result, n);
			std::vector<float> expected(n, 3);

			Assert::IsTrue(are_float_vectors_equal(expected, result));
			free_gpu_matrices(gpu_matrix_a, gpu_matrix_b, gpu_matrix_result, n);
		}
	};
}