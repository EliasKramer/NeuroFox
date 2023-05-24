#include "matrix.hpp"
#include <numeric>

bool matrix::is_initialized() const
{
	return host_data != nullptr && format.item_count() != 0;
}

void matrix::if_not_initialized_throw() const
{
	if (!is_initialized())
	{
		throw std::runtime_error("matrix not initialized");
	}
}

void matrix::if_not_owning_throw() const
{
	if (!owning_data)
	{
		throw std::runtime_error("matrix not owning data");
	}
}

bool matrix::is_device_mem_allocated() const
{
	return device_data != nullptr;
}

void matrix::if_gpu_not_allocated_throw() const
{
	if (!is_device_mem_allocated())
	{
		throw std::runtime_error("matrix not using gpu");
	}
}

void matrix::allocate_device_mem()
{
	if_not_initialized_throw();
	if (is_device_mem_allocated())
		throw std::runtime_error("gpu memory already allocated");

	cudaMalloc(&device_data, item_count() * sizeof(float));
	if_cuda_error_throw();
}

void matrix::copy_host_to_device()
{
	if_not_initialized_throw();
	if_gpu_not_allocated_throw();

	cudaMemcpy(
		device_data,
		host_data,
		item_count() * sizeof(float),
		cudaMemcpyHostToDevice);
	if_cuda_error_throw();
}
void matrix::copy_device_to_host()
{
	if_not_initialized_throw();
	if_gpu_not_allocated_throw();

	cudaMemcpy(
		host_data,
		device_data,
		item_count() * sizeof(float),
		cudaMemcpyDeviceToHost);
}

void matrix::if_cuda_error_throw() const
{
	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
	{
		std::string err_str = std::string(cudaGetErrorString(cuda_error));
		throw std::runtime_error(
			std::string("cuda error: ") +
			err_str);
	}
}

void matrix::check_for_valid_format() const
{
	if (format.item_count() == 0)
	{
		throw std::invalid_argument(
			"invalid format");
	}
}

void matrix::allocate_host_mem()
{
	if (host_data != nullptr)
	{
		throw std::runtime_error("cannot allocate if data will be overwritten");
	}
	if (item_count() == 0)
	{
		throw std::runtime_error("cannot allocate if item count is 0");
	}
	host_data = new float[item_count()];
	owning_data = true;
	set_all(0);
}

void matrix::set_own_host_data_from(const std::vector<float> src)
{
	if (src.empty())
	{
		throw std::runtime_error("cannot set data from empty vector");
	}

	if (src.size() != item_count())
	{
		throw std::runtime_error("cannot set data from vector of different size");
	}
	delete_data_if_owning();
	owning_data = true;

	std::copy(src.data(), src.data() + item_count(), this->host_data);

	//TODO gpu
	if (is_device_mem_allocated())
	{
		copy_host_to_device();
	}
}

void matrix::set_own_host_data_from(const matrix& src)
{
	src.if_not_initialized_throw();

	if (host_data == nullptr)
	{
		throw std::runtime_error("cannot set data if no memory is allocated");
	}
	if (!matrix::equal_format(*this, src))
	{
		throw std::runtime_error("cannot copy data from one matrix to another if they are not in the same format");
	}

	delete_data_if_owning();
	owning_data = true;

	std::copy(src.host_data, src.host_data + item_count(), this->host_data);

	//TODO gpu
	if (src.is_device_mem_allocated())
	{
		if (!is_device_mem_allocated())
		{
			allocate_device_mem();
		}
		copy_device_to_host();
	}
}

void matrix::delete_data_if_owning()
{
	if (owning_data)
	{
		if (host_data != nullptr)
		{
			delete[] host_data;
			host_data = nullptr;
		}
		if (device_data != nullptr)
		{
			cudaFree(device_data);
			if_cuda_error_throw();
			device_data = nullptr;
		}
		owning_data = false;
	}
}

float* matrix::get_ptr_layer(float* given_ptr, size_t depth_idx)
{
	if_not_initialized_throw();
	if (given_ptr == nullptr)
		throw std::invalid_argument("given_ptr is null");
	if (given_ptr != host_data && given_ptr != device_data)
		throw std::invalid_argument("given_ptr is not pointing to this matrix");

	return sub_ptr<float>(given_ptr, get_width() * get_height(), depth_idx);
}

float* matrix::get_ptr_row(float* given_ptr, size_t height_idx, size_t depth_idx)
{
	return get_ptr_layer(given_ptr, depth_idx) + height_idx * get_width();
}

float* matrix::get_ptr_item(float* given_ptr, size_t width_idx, size_t height_idx, size_t depth_idx)
{
	return get_ptr_row(given_ptr, height_idx, depth_idx) + width_idx;
}

matrix::matrix(
) :
	owning_data(false),
	host_data(nullptr),
	device_data(nullptr)
{}

matrix::matrix(
	vector3 given_format
) :
	format(given_format),
	owning_data(true),
	host_data(nullptr),
	device_data(nullptr)
{
	check_for_valid_format();
	allocate_host_mem();
}

matrix::matrix(
	vector3 given_format,
	const std::vector<float>& given_vector
) :
	matrix(given_format)
{
	set_own_host_data_from(given_vector);
}

matrix::matrix(const matrix& source)
	:matrix()
{
	if (source.is_initialized())
	{
		allocate_host_mem();
		set_own_host_data_from(source);
	}
}

matrix::~matrix()
{
	delete_data_if_owning();
}

matrix& matrix::operator=(const matrix& other)
{
	other.if_not_initialized_throw();
	//sets this matrix to the value of the other
	//by copying

	if (this != &other) {
		delete_data_if_owning();
		this->format = other.format;

		if (other.is_initialized() &&
			other.format.item_count() != 0)
		{
			allocate_host_mem();
			set_own_host_data_from(other);
			if (other.is_device_mem_allocated())
			{
				enable_gpu();
			}
		}
	}
	return *this;
}

void matrix::set_all(float value)
{
	if_not_initialized_throw();

	for (int i = 0; i < item_count(); i++)
	{
		host_data[i] = value;
	}
}

void matrix::apply_noise(float range)
{
	if_not_initialized_throw();

	for (int i = 0; i < item_count(); i++)
	{
		host_data[i] += random_float_incl(-range, range);
	}
}

void matrix::mutate(float range)
{
	if_not_initialized_throw();

	add_at_flat(
		random_idx((int)item_count()),
		random_float_incl(-range, range));
}

vector3 matrix::get_format() const
{
	return format;
}

size_t matrix::get_width() const
{
	return format.x;
}
size_t matrix::get_height() const
{
	return format.y;
}
size_t matrix::get_depth() const
{
	return format.z;
}

size_t matrix::item_count() const
{
	return format.item_count();
}

float matrix::get_at_flat(size_t idx) const
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}

	return host_data[idx];
}

void matrix::set_at_flat(size_t idx, float value)
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	host_data[idx] = value;
}

void matrix::add_at_flat(size_t idx, float value)
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	host_data[idx] += value;
}

float* matrix::get_device_ptr()
{
	return device_data;
}
const float* matrix::get_device_ptr_readonly() const
{
	return device_data;
}

float* matrix::get_device_ptr_layer(size_t depth_idx)
{
	if_not_initialized_throw();
	if_gpu_not_allocated_throw();

	return get_ptr_layer(device_data, depth_idx);
}

void matrix::enable_gpu()
{
	if_not_initialized_throw();

	if (!is_device_mem_allocated())
	{
		if (owning_data)
		{
			allocate_device_mem();
			copy_host_to_device();
		}
	}
}
/*
float* matrix::get_data()
{
	return host_data;
}

const float* matrix::get_data_readonly() const
{
	return host_data;
}
*/

void matrix::observe_row(matrix& m, size_t row_idx)
{
	observe_row(m, row_idx, 0);
}
void matrix::observe_row(matrix& m, size_t row_idx, size_t item_idx)
{
	m.if_not_initialized_throw();

	if (row_idx >= m.get_height())
	{
		throw std::invalid_argument("row_idx must be in range");
	}
	if (item_idx >= m.get_width())
	{
		throw std::invalid_argument("item_idx must be in range");
	}
	//if we have a matrix that is currently 3x3 it has an item count of 9
	//lets say the given matrix is 12x12
	//we select the current row of the given matrix
	//this row has 12 items
	//in order to set our 3x3 matrix, we need 9 items
	//in a row of 12 items, you have to start at either the 1st, 2nd, 3rd or 4th item
	//otherwise you would use more than 9 items
	// 
	//so the item_count() is 9
	//m.get_width() is 12
	//and now our item index has to be either 0, 1 or 2
	//lets test it. 
	//12 - 0 = 12 >= 9
	//12 - 1 = 11 >= 9
	//12 - 2 = 10 >= 9
	//12 - 3 = 9 >= 9
	//12 - 4 = 8 >= 9 (false)
	//so as soon as
	//12 - item_idx < 9 we throw an error
	if ((m.get_width() - row_idx) >
		item_count())
	{
		throw std::invalid_argument("the item count on this matrix must match the amount of items left in the given row");
	}

	delete_data_if_owning();

	host_data = m.get_ptr_item(m.host_data, item_idx, row_idx, 0);

	if (m.is_device_mem_allocated())
	{
		device_data = m.get_ptr_item(m.device_data, item_idx, row_idx, 0);
	}
	owning_data = false;
}

void matrix::set_row_from_matrix(const matrix& m, size_t row_idx)
{
	set_row_from_matrix(m, row_idx, 0);
}
void matrix::set_row_from_matrix(const matrix& m, size_t row_idx, size_t item_idx)
{
	m.if_not_initialized_throw();
	if_not_initialized_throw();
	if (!owning_data)
	{
		throw std::invalid_argument("this matrix must own its data");
	}
	if (!matrix::equal_format(format, m.format))
	{
		throw std::invalid_argument("the given matrix must have the same format as this matrix");
	}
	if (row_idx >= m.get_height())
	{
		throw std::invalid_argument("row_idx must be in range");
	}
	if (item_idx >= m.get_width())
	{
		throw std::invalid_argument("item_idx must be in range");
	}
	if ((get_width() - row_idx) >
		m.item_count())
	{
		throw std::invalid_argument("the item count on this matrix must match the amount of items left in the given row");
	}

	float* new_ptr = get_ptr_item(host_data, item_idx, row_idx, 0);
	memcpy(new_ptr, m.host_data, m.item_count() * sizeof(float));

	//TODO gpu
	if (is_device_mem_allocated())
	{
		copy_host_to_device();
	}
}

void matrix::set_at(vector3 pos, float value)
{
	if_not_initialized_throw();
	if_not_owning_throw();

	if (!pos.is_in_bounds(format))
		throw std::invalid_argument("pos must be in bounds");

	host_data[pos.get_index(format)] = value;

	//could be improved by only setting the value
	if (is_device_mem_allocated())
	{
		copy_host_to_device();
	}
}

void matrix::add_at(vector3 pos, float value)
{
	set_at(pos, get_at(pos) + value);
}

float matrix::get_at(vector3 pos) const
{
	if_not_initialized_throw();
	if (!pos.is_in_bounds(format))
		throw std::invalid_argument("pos must be in bounds");

	return host_data[pos.get_index(format)];
}

void matrix::dot_product(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (a.get_width() != b.get_height() || a.get_depth() != b.get_depth())
	{
		throw std::invalid_argument("dot product could not be performed. input matrices are in the wrong format");
	}
	if (result.get_width() != b.get_width() || result.get_height() != a.get_height() || result.get_depth() != a.get_depth())
	{
		throw std::invalid_argument("dot product could not be performed. result matrix is not the correct size");
	}

	for (int z = 0; z < result.get_depth(); z++)
	{
		for (int y = 0; y < result.get_height(); y++)
		{
			for (int x = 0; x < result.get_width(); x++)
			{
				float sum = 0;
				for (int i = 0; i < a.get_width(); i++)
				{
					sum += a.get_at(vector3(i, y, z)) * b.get_at(vector3(x, i, z));
				}
				result.set_at(vector3(x, y, z), sum);
			}
		}
	}
}

void matrix::dot_product_flat(const matrix& a, const matrix& flat, matrix& result_flat)
{
	a.if_not_initialized_throw();
	flat.if_not_initialized_throw();
	result_flat.if_not_initialized_throw();

	if (a.get_width() != flat.item_count() ||
		a.get_height() != result_flat.item_count() ||
		a.get_depth() != 1 ||
		result_flat.get_depth() != 1)
	{
		throw std::invalid_argument("dot product could not be performed. input matrices are in the wrong format");
	}

	for (int x = 0; x < a.get_width(); x++)
	{
		for (int y = 0; y < a.get_height(); y++)
		{
			result_flat.add_at_flat(y, a.get_at(vector3(x, y)) * flat.get_at_flat(x));
		}
	}
}

void matrix::add(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (a.get_width() != b.get_width() || a.get_height() != b.get_height() || a.get_depth() != b.get_depth())
	{
		throw std::invalid_argument("addition could not be performed. input matrices are in the wrong format");
	}
	if (result.get_width() != a.get_width() || result.get_height() != a.get_height() || result.get_depth() != a.get_depth())
	{
		throw std::invalid_argument("addition could not be performed. result matrix is not the correct size");
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		result.host_data[i] = a.host_data[i] + b.host_data[i];
	}
}

void matrix::add_flat(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (a.item_count() != b.item_count())
	{
		throw std::invalid_argument("addition could not be performed. input matrices are not the same size");
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		result.host_data[i] = a.host_data[i] + b.host_data[i];
	}
}

void matrix::subtract(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();
	result.if_not_owning_throw();

	if (!equal_format(a, b) ||
		!equal_format(b, result) ||
		!equal_format(result, a))
	{
		throw std::invalid_argument("subtraction could not be performed. input matrices are in the wrong format");
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		result.host_data[i] = a.host_data[i] - b.host_data[i];
	}
}

bool matrix::are_equal(const matrix& a, const matrix& b)
{
	return are_equal(a, b, FLOAT_TOLERANCE);
}

bool matrix::are_equal(const matrix& a, const matrix& b, float tolerance)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();

	if (!equal_format(a, b))
	{
		return false;
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		if (std::abs(a.host_data[i] - b.host_data[i]) > tolerance)
		{
			return false;
		}
	}
	return true;
}

bool matrix::equal_format(const matrix& a, const matrix& b)
{
	return vector3::are_equal(a.format, b.format);
}

void matrix::valid_cross_correlation(
	const matrix& input,
	const std::vector<matrix>& kernels,
	matrix& output,
	int stride)
{
	input.if_not_initialized_throw();
	output.if_not_initialized_throw();
	output.if_not_owning_throw();
	for (const auto& curr_kernel : kernels)
	{
		curr_kernel.if_not_initialized_throw();
	}

	//this only works with a stride of one
	const size_t input_size = input.get_width();
	const size_t kernel_size = kernels[0].get_width();
	const size_t output_size = output.get_width();
	const float expected_output_size = ((input_size - kernel_size) / (float)stride) + 1;

	if ((float)output_size != expected_output_size)
	{
		throw std::invalid_argument("cross correlation could not be performed. output matrix is not the correct size");
	}
	if (input.get_depth() != kernels[0].get_depth())
	{
		throw std::invalid_argument("cross correlation could not be performed. input matrices are in the wrong format");
	}

	output.set_all(0);

	//iterate over each
	for (int z = 0; z < output.get_depth(); z++)
	{
		for (int y = 0; y < output.get_height(); y++)
		{
			for (int x = 0; x < output.get_width(); x++)
			{
				//iterate over the kernel and input.
				//the overlaying values are multiplied and added to the output
				//input		kernel
				//+--+--+	+--+--+	  +--+--+
				//|1 |2 |	|2 |3 |	  |2 |6 |
				//+--+--+ * +--+--+ = +--+--+ = 2 + 6 + 12 + 20 = 40 
				//|3 |4 |	|4 |5 |	  |12|20|
				//+--+--+	+--+--+	  +--+--+
				//
				//40 is the sum

				float sum = 0;
				for (int curr_depth = 0; curr_depth < kernels[0].get_depth(); curr_depth++)
				{
					for (int i = 0; i < kernel_size; i++)
					{
						for (int j = 0; j < kernel_size; j++)
						{
							sum +=
								input.get_at(
									vector3(
										x * stride + i,
										y * stride + j,
										curr_depth)) *
								kernels[z].get_at(
									vector3(
										i,
										j,
										curr_depth));
						}
					}
				}
				//if we do this, the output of all depths will be added together
				output.add_at(vector3(x, y, z), sum);
			}
		}
	}
}

void matrix::scalar_multiplication(float a)
{
	if_not_initialized_throw();
	if_not_owning_throw();

	for (int i = 0; i < item_count(); i++)
	{
		host_data[i] *= a;
	}
}

void matrix::apply_activation_function(e_activation_t activation_fn)
{
	if_not_initialized_throw();
	if_not_owning_throw();

	for (int i = 0; i < item_count(); i++)
	{
		host_data[i] = ACTIVATION[activation_fn](host_data[i]);
	}
}

std::string matrix::get_string() const
{
	if_not_initialized_throw();

	std::string ret_val = "";

	for (int z = 0; z < get_depth(); z++)
	{
		for (int y = 0; y < get_height(); y++)
		{
			for (int x = 0; x < get_width(); x++)
			{
				ret_val += std::to_string(get_at(vector3(x, y, z))) + " ";
			}
			ret_val += "\n";
		}
		ret_val += "\n";
	}

	return ret_val;
}