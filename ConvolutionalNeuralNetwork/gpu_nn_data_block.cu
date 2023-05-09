#include "gpu_nn_data_block.cuh"

gpu_nn_data_block::gpu_nn_data_block(
	size_t num_of_blocks, 
	size_t num_of_data,
	size_t num_of_label_data): 
	num_of_blocks(num_of_blocks),
	num_of_label_data(num_of_label_data),
	num_of_data(num_of_data)
{
	if (num_of_blocks == 0 || num_of_data == 0)
		throw std::runtime_error("could not create gpu_nn_data_block");
	
	data = std::make_unique<gpu_memory<float>>(num_of_blocks * (num_of_data + num_of_label_data));
}

size_t gpu_nn_data_block::get_num_of_blocks() const
{
	return num_of_blocks;
}

size_t gpu_nn_data_block::get_num_of_label_data() const
{
	return num_of_label_data;
}

size_t gpu_nn_data_block::get_num_of_data() const
{
	return num_of_data;
}

float* gpu_nn_data_block::get_gpu_data_ptr(int idx)
{
	if(idx >= num_of_blocks || idx < 0)
		throw std::runtime_error("index out of bounds");

	return gpu_sub_ptr(*data.get(), num_of_data + num_of_label_data, idx);
}

float* gpu_nn_data_block::get_gpu_label_ptr(int idx)
{
	if (idx >= num_of_blocks || idx < 0)
		throw std::runtime_error("index out of bounds");
	if (num_of_label_data == 0)
		throw std::runtime_error("this block has no label data");

	return gpu_sub_ptr(*data.get(), num_of_data + num_of_label_data, idx) + (num_of_data);
}

void gpu_nn_data_block::set_data(int idx, const std::vector<float>& data)
{
	if (data.size() != num_of_data)
		throw std::runtime_error("could net set gpu data in block");

	cudaMemcpy(get_gpu_data_ptr(idx), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess)
		throw std::runtime_error("could net set gpu data in block");
}

void gpu_nn_data_block::set_label_data(int idx, const std::vector<float>& data)
{
	if (data.size() != num_of_label_data)
		throw std::runtime_error("could net set gpu data in block");

	cudaMemcpy(get_gpu_label_ptr(idx), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess)
		throw std::runtime_error("could net set gpu data in block");
}
