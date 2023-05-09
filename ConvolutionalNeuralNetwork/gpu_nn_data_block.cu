#include "gpu_nn_data_block.cuh"

gpu_nn_data_block::gpu_nn_data_block(
	size_t num_of_blocks, 
	size_t num_of_data,
	size_t num_of_label_data): 
	num_of_blocks(num_of_blocks),
	num_of_label_data(num_of_label_data),
	num_of_data(num_of_data),
	data(gpu_memory<float>(num_of_blocks * (num_of_data + num_of_label_data)))
{}

size_t gpu_nn_data_block::get_num_of_blocks() const
{
	return size_t();
}

size_t gpu_nn_data_block::get_num_of_label_data() const
{
	return size_t();
}

size_t gpu_nn_data_block::get_num_of_data() const
{
	return size_t();
}

float* gpu_nn_data_block::get_gpu_data_ptr(int idx)
{
	if(idx >= num_of_blocks)
		throw std::exception("could net get gpu data in block");

	return gpu_sub_ptr(data, num_of_data + num_of_label_data, idx);
}

float* gpu_nn_data_block::get_gpu_label_ptr(int idx)
{
	if (idx >= num_of_blocks)
		throw std::exception("could net get gpu data in block");

	return gpu_sub_ptr(data, num_of_data + num_of_label_data, idx) + (num_of_data);
}

void gpu_nn_data_block::set_data(int idx, const std::vector<float>& data)
{
	if (data.size() != num_of_data)
		throw std::exception("could net set gpu data in block");

	cudaMemcpy(get_gpu_data_ptr(idx), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess)
		throw std::exception("could net set gpu data in block");
}

void gpu_nn_data_block::set_label_data(int idx, const std::vector<float>& data)
{
	if (data.size() != num_of_label_data)
		throw std::exception("could net set gpu data in block");

	cudaMemcpy(get_gpu_label_ptr(idx), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess)
		throw std::exception("could net set gpu data in block");
}
