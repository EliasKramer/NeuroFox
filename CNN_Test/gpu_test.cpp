#include "CppUnitTest.h"
#include "../ConvolutionalNeuralNetwork/gpu_math.cuh"
#include "../ConvolutionalNeuralNetwork/gpu_nn_data_block.cuh"
#include "../ConvolutionalNeuralNetwork/util.hpp"
#include "test_util.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CNNTest
{
	TEST_CLASS(cuda_test)
	{
	public:
		std::vector<float> get_gpu_values(const float* gpu_ptr, size_t n)
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
			gpu_matrix gpu_mem(2, 1, 1);
			gpu_mem.set_all((float)0xC00FFEE);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.get_gpu_memory_readonly(), gpu_mem.item_count());
			std::vector<float> expected_values(gpu_mem.item_count(), (float)0xC00FFEE);
			Assert::IsTrue(float_vectors_equal(expected_values, gpu_values));
		}
		TEST_METHOD(gpu_memory_destructor_test)
		{
			gpu_matrix* gpu_mem = new gpu_matrix(2, 1, 1);

			float* gpu_ptr = gpu_mem->get_gpu_memory();
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

			gpu_matrix gpu_mem(m, true);

			std::vector<float> gpu_values = get_gpu_values(gpu_mem.get_gpu_memory(), m.item_count());
			std::vector<float> expected_values(m.item_count(), (float)0xDEADBEEF);
			Assert::IsTrue(float_vectors_equal(expected_values, gpu_values));
		}
		/*
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

			gpu_matrix gpu_mem(m, true);

			float* gpu_ptr = sub_ptr<float>(gpu_mem.get_gpu_memory(), 4, 1);

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
		*/
		TEST_METHOD(add_gpu_test)
		{
			int n = 1000000;
			gpu_matrix gpu_mem_a(n, 1, 1);
			gpu_matrix gpu_mem_b(n, 1, 1);
			gpu_matrix gpu_mem_result(n, 1, 1);

			gpu_mem_a.set_all(1);
			gpu_mem_b.set_all(2);
			gpu_mem_result.set_all(0);

			gpu_add(gpu_mem_a, gpu_mem_b, gpu_mem_result);

			matrix result = *gpu_mem_result.to_cpu().get();
			matrix expected(n, 1, 1);
			expected.set_all(3);
			Assert::IsTrue(matrix::are_equal(expected, result));
		}
		TEST_METHOD(gpu_activation_relu_test)
		{
			matrix m(1, 3, 1);
			m.set_at(0, 0, 1);
			m.set_at(0, 1, 0);
			m.set_at(0, 2, -3);

			gpu_matrix gpu_mem(m, true);
			GPU_ACTIVATION[relu_fn](gpu_mem);

			std::vector<float> expected_v{ 1, 0, 0 };
			matrix expected = matrix(1, 3, 1, expected_v);
			Assert::IsTrue(matrix::are_equal(expected, *gpu_mem.to_cpu().get()));
		}
		TEST_METHOD(gpu_activation_sigmoid_test)
		{
			matrix m(1, 3, 1);
			m.set_at(0, 0, 1);
			m.set_at(0, 1, 0);
			m.set_at(0, 2, -3);

			gpu_matrix gpu_mem(m, true);
			GPU_ACTIVATION[sigmoid_fn](gpu_mem);

			matrix result = *gpu_mem.to_cpu().get();
			std::vector<float> expected_v = {
				ACTIVATION[sigmoid_fn](1),
				ACTIVATION[sigmoid_fn](0),
				ACTIVATION[sigmoid_fn](-3)
			};
			matrix expected = matrix(1, 3, 1, expected_v);
			Assert::IsTrue(matrix::are_equal(expected, result));
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

			gpu_matrix gpu_input(m_input, true);
			gpu_matrix gpu_weights(m_weights, true);
			gpu_matrix gpu_activations(m_activations, true);

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

			matrix result = *gpu_activations.to_cpu().get();
			Assert::IsTrue(matrix::are_equal(expected_activations, result));
		}
		TEST_METHOD(gpu_valid_cross_correlation_test_1)
		{
			std::vector<float> input_data = {
				1, 2, 3,
				4, 5, 6,
				7, 8, 9
			};
			matrix input(3, 3, 1, input_data);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4
			};
			matrix kernel(2, 2, 1, kernel_data);

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

			gpu_matrix gpu_input(input, true);
			gpu_matrix gpu_result(2, 2, 1);

			std::vector<std::unique_ptr<gpu_matrix>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel, true));

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

			Assert::IsTrue(matrix::are_equal(expected, *gpu_result.to_cpu().get()));
		}
		TEST_METHOD(gpu_valid_cross_correlation_test_2)
		{
			std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7
			};
			matrix input(4, 4, 1, input_data);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4
			};
			matrix kernel(2, 2, 1, kernel_data);

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

			gpu_matrix gpu_input(input, true);
			gpu_matrix gpu_result(2, 2, 1);

			std::vector<std::unique_ptr<gpu_matrix>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel, true));

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

			Assert::IsTrue(matrix::are_equal(expected, *gpu_result.to_cpu().get()));
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
			matrix input(4, 4, 2, input_data);

			std::vector<float> kernel_data = {
				1, 2,
				3, 4,

				1, 2,
				3, 4
			};
			matrix kernel(2, 2, 2, kernel_data);

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

			gpu_matrix gpu_input(input, true);
			gpu_matrix gpu_result(2, 2, 1);

			std::vector<std::unique_ptr<gpu_matrix>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel, true));

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

			Assert::IsTrue(matrix::are_equal(expected, *gpu_result.to_cpu().get()));
		}
		TEST_METHOD(gpu_valid_cross_correlation_output_depth_2_test)
		{
			std::vector<float> input_data = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 1, 2, 3,
				4, 5, 6, 7
			};
			matrix input(4, 4, 1, input_data);

			std::vector<float> kernel_data1 = {
				1, 2,
				3, 4
			};
			std::vector<float> kernel_data2 = {
				5, 6,
				7, 8
			};
			matrix kernel1(2, 2, 1, kernel_data1);
			matrix kernel2(2, 2, 1, kernel_data2);

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

			gpu_matrix gpu_input(input, true);
			gpu_matrix gpu_result(2, 2, 2);

			std::vector<std::unique_ptr<gpu_matrix>> gpu_kernel_weights;
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel1, true));
			gpu_kernel_weights.emplace_back(std::make_unique<gpu_matrix>(kernel2, true));

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

			Assert::IsTrue(matrix::are_equal(expected, *gpu_result.to_cpu().get()));
		}
		TEST_METHOD(set_at_test)
		{
			gpu_matrix m(3, 1, 1);
			m.set_all((float)0xBEEFCACE);
			m.set_at(1, 0, 0, (float)0xDEADBEEF);

			std::vector<float> expected = {
				(float)0xBEEFCACE,
				(float)0xDEADBEEF,
				(float)0xBEEFCACE
			};

			Assert::IsTrue(
				matrix::are_equal(
					matrix(3, 1, 1, expected),
					*m.to_cpu().get()));
		}
		TEST_METHOD(test_set_values_and_convert_to_cpu)
		{
			gpu_matrix gpu1(2, 2, 1);
			gpu1.set_all(1.0f);

			gpu_matrix gpu2(2, 2, 1);
			gpu2.set_values_gpu(gpu1);

			std::vector<float> expected = {
				1.0f, 1.0f,
				1.0f, 1.0f
			};

			Assert::IsTrue(matrix::are_equal(matrix(2, 2, 1, expected), *gpu2.to_cpu().get()));
		}
		TEST_METHOD(test_set_values_from_gpu)
		{
			gpu_matrix gpu1(2, 2, 1);
			gpu1.set_all(1.0f);

			gpu_matrix gpu2(2, 2, 1);
			gpu2.set_values_gpu(gpu1);

			//cloned matrix should not be influenced by the original
			//change original
			gpu1.set_at(0, 0, 0, 2.0f);
			//create an identical cpu matrix
			matrix expected = matrix(2, 2, 1);
			//set all values to 1.0f (like the inital copied values)
			expected.set_all(1.0f);
			Assert::IsTrue(matrix::are_equal(expected, *gpu2.to_cpu().get(), FLOAT_TOLERANCE));

			//cloned matrix should not influence the original
			//change cloned
			gpu2.set_at(1, 0, 0, 3.0f);
			//the cpu matrix is currently all ones.
			//first we have to set the value at 0,0,0 to 2.0f (the change that we tested before)
			expected.set_at(0, 0, 0, 2.0f);
			//the first one on the cloned matrix should be 3.0f, but not the original
			Assert::IsTrue(matrix::are_equal(expected, *gpu1.to_cpu().get(), FLOAT_TOLERANCE));
		}
		//test clone and set values are essentially the same thing.
		//i like to test them separately, in case the implementation changes
		TEST_METHOD(test_clone)
		{
			gpu_matrix gpu1(2, 2, 1);
			gpu1.set_all(1.0f);

			std::unique_ptr<gpu_matrix> gpu2 = gpu1.clone();

			//cloned matrix should not be influenced by the original
			//change original
			gpu1.set_at(0, 0, 0, 2.0f);
			//create an identical cpu matrix
			matrix expected = matrix(2, 2, 1);
			//set all values to 1.0f (like the inital copied values)
			expected.set_all(1.0f);
			Assert::IsTrue(matrix::are_equal(expected, *gpu2->to_cpu().get(), FLOAT_TOLERANCE));

			//cloned matrix should not influence the original
			//change cloned
			gpu2->set_at(1, 0, 0, 3.0f);
			//the cpu matrix is currently all ones.
			//first we have to set the value at 0,0,0 to 2.0f (the change that we tested before)
			expected.set_at(0, 0, 0, 2.0f);
			//the first one on the cloned matrix should be 3.0f, but not the original
			Assert::IsTrue(matrix::are_equal(expected, *gpu1.to_cpu().get(), FLOAT_TOLERANCE));
		}

		/*
		TEST_METHOD(data_block_test_1)
		{
			gpu_nn_data_block block = gpu_nn_data_block(3, 2, 1);
			block.set_data(0, std::vector<float>{1, 2});
			block.set_data(1, std::vector<float>{3, 4});
			block.set_data(2, std::vector<float>{5, 6});

			block.set_label_data(0, std::vector<float>{7});
			block.set_label_data(1, std::vector<float>{8});
			block.set_label_data(2, std::vector<float>{9});

			std::vector<float> gpu_values = get_gpu_values(block.get_get_gpu_memory(0), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {1, 2}));
			gpu_values = get_gpu_values(block.get_get_gpu_memory(1), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {3, 4}));
			gpu_values = get_gpu_values(block.get_get_gpu_memory(2), 2);
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

			std::vector<float> gpu_values = get_gpu_values(block.get_get_gpu_memory(0), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {1, 2}));
			gpu_values = get_gpu_values(block.get_get_gpu_memory(1), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {3, 4}));

			gpu_values = get_gpu_values(block.get_gpu_label_ptr(0), 3);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {5, 6, 7}));
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(1), 3);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {8, 9, 1}));
		}
		TEST_METHOD(data_block_constructor_test)
		{
			gpu_nn_data_block block = gpu_nn_data_block(2, 2, 3);
			//you can make blocks without labels
			block = gpu_nn_data_block(2, 2, 0);

			try
			{
				block = gpu_nn_data_block(0, 2, 3);
				Assert::Fail();
			}
			catch (std::runtime_error& e) {
				Assert::AreEqual(e.what(), "could not create gpu_nn_data_block");
			}
			try
			{
				block = gpu_nn_data_block(2, 0, 3);
				Assert::Fail();
			}
			catch (std::runtime_error& e) {
				Assert::AreEqual(e.what(), "could not create gpu_nn_data_block");
			}
		}
		TEST_METHOD(data_block_wrong_indexing_test)
		{
			gpu_nn_data_block block = gpu_nn_data_block(2, 2, 3);

			try
			{
				block.set_data(3, std::vector<float>{1, 2});
				Assert::Fail();
			}
			catch (std::runtime_error e)
			{
				Assert::AreEqual(e.what(), "index out of bounds");
			}
			try {
				block.get_get_gpu_memory(-1);
				Assert::Fail();
			}
			catch (std::runtime_error e)
			{
				Assert::AreEqual(e.what(), "index out of bounds");
			}
		}
		TEST_METHOD(data_block_no_label_data)
		{
			gpu_nn_data_block block = gpu_nn_data_block(2, 2, 0);

			try
			{
				block.get_gpu_label_ptr(0);
				Assert::Fail();
			}
			catch (std::runtime_error e)
			{
				Assert::AreEqual(e.what(), "this block has no label data");
			}
		}
		TEST_METHOD(set_data_with_cpu_nn_data)
		{
			std::vector data1 = std::vector<float>{ 1, 2 };
			std::vector label1 = std::vector<float>{ 3 };
			std::vector data2 = std::vector<float>{ 4, 5 };
			std::vector label2 = std::vector<float>{ 6 };

			nn_data nn_data1 = nn_data(matrix(data1, 1, 2, 1), matrix(label1, 1, 1, 1));
			nn_data nn_data2 = nn_data(matrix(data2, 1, 2, 1), matrix(label2, 1, 1, 1));

			std::vector<nn_data> data_vec = std::vector<nn_data>{ nn_data1, nn_data2 };

			gpu_nn_data_block block = gpu_nn_data_block(2, 2, 1);
			block.set_data(data_vec.cbegin(), data_vec.cend());

			std::vector<float> gpu_values = get_gpu_values(block.get_get_gpu_memory(0), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {1, 2}));
			gpu_values = get_gpu_values(block.get_get_gpu_memory(1), 2);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {4, 5}));

			gpu_values = get_gpu_values(block.get_gpu_label_ptr(0), 1);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {3}));
			gpu_values = get_gpu_values(block.get_gpu_label_ptr(1), 1);
			Assert::IsTrue(float_vectors_equal(gpu_values, std::vector<float> {6}));
		}
		*/
	};
}