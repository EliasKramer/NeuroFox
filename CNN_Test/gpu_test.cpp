#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/gpu_math.cuh"
#include "../ConvolutionalNeuralNetwork/gpu_nn_data_block.cuh"
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
			gpu_memory<float>* gpu_mem = new gpu_memory<float>(2);

			float* gpu_ptr = gpu_mem->gpu_data_ptr();
			delete gpu_mem;

			std::vector<float> cpu_data(2);
			cudaMemcpy(
				cpu_data.data(),
				gpu_ptr, 2 * sizeof(float),
				cudaMemcpyDeviceToHost);

			// Check if memory has been correctly freed
			Assert::AreNotEqual((int)cudaSuccess, (int)cudaGetLastError());
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
		TEST_METHOD(sub_ptr_test)
		{
			// depth 0
			// +-+-+
			// |0|2|
			// +-+-+
			// |1|3|
			// +-+-+
			// depth 1
			// +-+-+
			// |4|6|
			// +-+-+
			// |5|7|
			// +-+-+
			matrix m(2, 2, 2);
			for (float i = 0; i < 8; i++)
			{
				m.flat()[i] = i;
			}

			gpu_memory<float> gpu_mem(m);

			float* gpu_ptr = gpu_sub_ptr(gpu_mem, 4, 1);

			std::vector<float> gpu_data = get_gpu_values(gpu_ptr, 4);
			// +-+-+
			// |4|6|
			// +-+-+
			// |5|7|
			// +-+-+
			std::vector<float> expected_data(4);
			for (float i = 0; i < 4; i++)
			{
				expected_data[i] = i + 4;
			}
			matrix expected(expected_data, 2, 2, 1);
			matrix gpu_matrix(gpu_data, 2, 2, 1);

			Assert::IsTrue(matrix::are_equal(expected, gpu_matrix));
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
		TEST_METHOD(gpu_valid_cross_correlation_test_1)
		{
			std::vector<float> input_data = {
				1, 2, 3,
				4, 5, 6,
				7, 8, 9
			};
			matrix input(input_data, 3, 3, 1);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4
			};
			matrix kernel(kernel_data, 2, 2, 1);

			matrix expected(2, 2, 1);
			expected.set_at(0, 0,
				1 * 1 + 2 * 2 +
				3 * 4 + 4 * 5);
			expected.set_at(1, 0,
				1 * 2 + 2 * 3 +
				3 * 5 + 4 * 6);
			expected.set_at(0, 1,
				1 * 4 + 2 * 5 +
				3 * 7 + 4 * 8);
			expected.set_at(1, 1,
				1 * 5 + 2 * 6 +
				3 * 8 + 4 * 9);

			matrix double_check_expected(2, 2, 1);
			std::vector<matrix> cpu_kernels = { kernel };
			matrix::valid_cross_correlation(input, cpu_kernels, double_check_expected, 1);

			Assert::IsTrue(matrix::are_equal(expected, double_check_expected));

			gpu_memory<float> gpu_input(input);
			gpu_memory<float> gpu_result(4);

			std::vector<std::unique_ptr<gpu_memory<float>>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel));

			gpu_valid_cross_correlation(
				gpu_input,
				gpu_kernel_weights,
				gpu_result,
				input.get_width(),
				input.get_depth(),
				kernel.get_width(),
				gpu_kernel_weights.size(),
				1, //stride 
				2); //output width

			std::vector<float> result = *gpu_result.to_cpu().get();
			matrix result_matrix = matrix(result, 2, 2, 1);
			Assert::IsTrue(matrix::are_equal(expected, result_matrix));
		}
		TEST_METHOD(gpu_valid_cross_correlation_test_2)
		{
			std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7
			};
			matrix input(input_data, 4, 4, 1);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4
			};
			matrix kernel(kernel_data, 2, 2, 1);

			matrix expected(2, 2, 1);
			expected.set_at(0, 0,
				1 * 1 + 2 * 2 +
				3 * 5 + 4 * 6);
			expected.set_at(1, 0,
				1 * 3 + 2 * 4 +
				3 * 7 + 4 * 8);
			expected.set_at(0, 1,
				1 * 9 + 2 * 1 +
				3 * 4 + 4 * 5);
			expected.set_at(1, 1,
				1 * 2 + 2 * 3 +
				3 * 6 + 4 * 7);

			matrix double_check_expected(2, 2, 1);
			std::vector<matrix> cpu_kernels = { kernel };
			matrix::valid_cross_correlation(input, cpu_kernels, double_check_expected, 2);

			Assert::IsTrue(matrix::are_equal(expected, double_check_expected));

			gpu_memory<float> gpu_input(input);
			gpu_memory<float> gpu_result(4);


			std::vector<std::unique_ptr<gpu_memory<float>>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel));

			gpu_valid_cross_correlation(
				gpu_input,
				gpu_kernel_weights,
				gpu_result,
				input.get_width(),
				input.get_depth(),
				kernel.get_width(),
				gpu_kernel_weights.size(),
				2, //stride 
				2); //output width

			std::vector<float> result = *gpu_result.to_cpu().get();
			matrix result_matrix = matrix(result, 2, 2, 1);
			Assert::IsTrue(matrix::are_equal(expected, result_matrix));
		}
		TEST_METHOD(gpu_valid_cross_correlation_input_depth_2_test)
		{
			std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7,

				8, 9, 1, 2,
				3, 4, 5, 6,
				7, 8, 9, 1,
				2, 3, 4, 5
			};
			matrix input(input_data, 4, 4, 2);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4,

				1, 2,
				3, 4
			};
			matrix kernel(kernel_data, 2, 2, 2);

			matrix expected(2, 2, 1);
			expected.set_at(0, 0, 0,
				1 * 1 + 2 * 2 +
				3 * 5 + 4 * 6);
			expected.set_at(1, 0, 0,
				1 * 3 + 2 * 4 +
				3 * 7 + 4 * 8);
			expected.set_at(0, 1, 0,
				1 * 9 + 2 * 1 +
				3 * 4 + 4 * 5);
			expected.set_at(1, 1, 0,
				1 * 2 + 2 * 3 +
				3 * 6 + 4 * 7);

			expected.add_at(0, 0, 0,
				1 * 8 + 2 * 9 +
				3 * 3 + 4 * 4);
			expected.add_at(1, 0, 0,
				1 * 1 + 2 * 2 +
				3 * 5 + 4 * 6);
			expected.add_at(0, 1, 0,
				1 * 7 + 2 * 8 +
				3 * 2 + 4 * 3);
			expected.add_at(1, 1, 0,
				1 * 9 + 2 * 1 +
				3 * 4 + 4 * 5);

			matrix double_check_expected(2, 2, 1);
			std::vector<matrix> cpu_kernels = { kernel };
			matrix::valid_cross_correlation(input, cpu_kernels, double_check_expected, 2);

			Assert::IsTrue(matrix::are_equal(expected, double_check_expected));

			gpu_memory<float> gpu_input(input);
			gpu_memory<float> gpu_result(4);

			std::vector<std::unique_ptr<gpu_memory<float>>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel));

			gpu_valid_cross_correlation(
				gpu_input,
				gpu_kernel_weights,
				gpu_result,
				input.get_width(),
				input.get_depth(),
				kernel.get_width(),
				gpu_kernel_weights.size(),
				2, //stride 
				2); //output width

			std::vector<float> result = *gpu_result.to_cpu().get();
			matrix result_matrix = matrix(result, 2, 2, 1);
			Assert::IsTrue(matrix::are_equal(expected, result_matrix));
		}
		TEST_METHOD(gpu_valid_cross_correlation_output_depth_2_test)
		{
			std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7
			};
			matrix input(input_data, 4, 4, 1);

			std::vector<float> kernel_data1 = {
				1, 2,
				3, 4
			};
			std::vector<float> kernel_data2 = {
				5, 6,
				7, 8
			};
			matrix kernel1(kernel_data1, 2, 2, 1);
			matrix kernel2(kernel_data2, 2, 2, 1);

			matrix expected(2, 2, 2);
			expected.set_at(0, 0, 0,
				1 * 1 + 2 * 2 +
				3 * 5 + 4 * 6);
			expected.set_at(1, 0, 0,
				1 * 3 + 2 * 4 +
				3 * 7 + 4 * 8);
			expected.set_at(0, 1, 0,
				1 * 9 + 2 * 1 +
				3 * 4 + 4 * 5);
			expected.set_at(1, 1, 0,
				1 * 2 + 2 * 3 +
				3 * 6 + 4 * 7);

			expected.set_at(0, 0, 1,
				5 * 1 + 6 * 2 +
				7 * 5 + 8 * 6);
			expected.set_at(1, 0, 1,
				5 * 3 + 6 * 4 +
				7 * 7 + 8 * 8);
			expected.set_at(0, 1, 1,
				5 * 9 + 6 * 1 +
				7 * 4 + 8 * 5);
			expected.set_at(1, 1, 1,
				5 * 2 + 6 * 3 +
				7 * 6 + 8 * 7);

			matrix double_check_expected(2, 2, 2);
			std::vector<matrix> cpu_kernels = { kernel1, kernel2 };
			matrix::valid_cross_correlation(input, cpu_kernels, double_check_expected, 2);

			Assert::IsTrue(matrix::are_equal(expected, double_check_expected));

			gpu_memory<float> gpu_input(input);
			gpu_memory<float> gpu_result(8);

			std::vector<std::unique_ptr<gpu_memory<float>>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel1));
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(kernel2));

			gpu_valid_cross_correlation(
				gpu_input,
				gpu_kernel_weights,
				gpu_result,
				input.get_width(),
				input.get_depth(),
				kernel1.get_width(),
				gpu_kernel_weights.size(),
				2, //stride 
				2); //output width

			std::vector<float> result = *gpu_result.to_cpu().get();
			matrix result_matrix = matrix(result, 2, 2, 2);
			Assert::IsTrue(matrix::are_equal(expected, result_matrix));
		}
		TEST_METHOD(data_block_test_1)
		{
			gpu_nn_data_block block = gpu_nn_data_block(3, 2, 1);
			block.set_data(0, std::vector<float>{1, 2});
			block.set_data(1, std::vector<float>{3, 4});
			block.set_data(2, std::vector<float>{5, 6});

			block.set_label_data(0, std::vector<float>{7});
			block.set_label_data(1, std::vector<float>{8});
			block.set_label_data(2, std::vector<float>{9});

			std::vector<float> gpu_values = get_gpu_values(block.get_gpu_data_ptr(0), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {1, 2}));
			gpu_values = get_gpu_values(block.get_gpu_data_ptr(1), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {3, 4}));
			gpu_values = get_gpu_values(block.get_gpu_data_ptr(2), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {5, 6}));

			gpu_values = get_gpu_values(block.get_gpu_label_ptr(0), 1);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {7}));
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(1), 1);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {8}));
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(2), 1);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {9}));
		}
		TEST_METHOD(data_block_test_2)
		{
			gpu_nn_data_block block = gpu_nn_data_block(2, 2, 3);
			block.set_data(0, std::vector<float>{1, 2});
			block.set_data(1, std::vector<float>{3, 4});

			block.set_label_data(0, std::vector<float>{5, 6, 7});
			block.set_label_data(1, std::vector<float>{8, 9, 1});

			std::vector<float> gpu_values = get_gpu_values(block.get_gpu_data_ptr(0), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {1, 2}));
			gpu_values = get_gpu_values(block.get_gpu_data_ptr(1), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {3, 4}));
			
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(0), 3);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {5, 6, 7}));
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(1), 3);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {8, 9, 1}));
		}
	};
}