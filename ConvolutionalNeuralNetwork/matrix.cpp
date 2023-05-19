#include "matrix.hpp"
#include <numeric>

size_t matrix::get_idx(size_t x, size_t y, size_t z) const
{
	if (x >= width || y >= height || z >= depth)
		throw std::invalid_argument("index out of bounds");

	return x + y * width + z * width * height;
}

void matrix::if_not_initialized_throw() const
{
	if (data == nullptr ||
		width == 0 ||
		height == 0 ||
		depth == 0)
	{
		throw std::runtime_error("matrix not initialized");
	}
}

void matrix::check_for_valid_format() const
{
	if (width == 0 || height == 0 || depth == 0)
	{
		throw std::invalid_argument(
			"invalid format");
	}
}

void matrix::allocate_mem()
{
	if (data != nullptr)
	{
		throw std::runtime_error("cannot allocate if data will be overwritten");
	}
	if (item_count() == 0)
	{
		throw std::runtime_error("cannot allocate if item count is 0");
	}
	data = new float[item_count()];
	owning_data = true;
	set_all(0);
}

void matrix::set_own_data_from(const float* src)
{
	if_not_initialized_throw();

	if (src == nullptr)
		throw std::runtime_error("src is null");
	if (!owning_data)
		throw std::runtime_error("cannot set data if not owned");

	std::copy(src, src + item_count(), this->data);
}

void matrix::set_own_data_from(const matrix& src)
{
	if_not_initialized_throw();
	src.if_not_initialized_throw();

	if (src.item_count() != item_count())
		throw std::runtime_error("cannot copy data if not the same size");

	set_own_data_from(src.data);
}

void matrix::delete_data_if_owning()
{
	if (owning_data && data != nullptr)
	{
		delete[] data;
		data = nullptr;
		owning_data = false;
	}
}

matrix::matrix(
) :
	width(0),
	height(0),
	depth(0),
	owning_data(false),
	data(nullptr)
{}

matrix::matrix(
	size_t width,
	size_t height,
	size_t depth
) :
	matrix(
		width,
		height,
		depth,
		nullptr,
		false)
{}

matrix::matrix(
	size_t width,
	size_t height,
	size_t depth,
	float* given_ptr,
	bool copy
) :
	width(width),
	height(height),
	depth(depth),
	owning_data(copy),
	data(given_ptr)
{
	check_for_valid_format();
	allocate_mem();
	if (copy)
	{
		set_own_data_from(given_ptr);
	}
}

matrix::matrix(
	size_t width,
	size_t height,
	size_t depth,
	const std::vector<float>& given_vector
) :
	matrix(
		width,
		height,
		depth)
{
	owning_data = true;
	set_own_data_from(given_vector.data());
}

matrix::matrix(const matrix& source)
	:matrix()
{
	initialize_format(source);
	set_own_data_from(source.get_data_readonly());
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
		initialize_format(other);
		set_own_data_from(other.data);
	}
	return *this;
}

void matrix::initialize_format(size_t width, size_t height, size_t depth)
{
	this->width = width;
	this->height = height;
	this->depth = depth;

	check_for_valid_format();
	delete_data_if_owning();
	allocate_mem();
}

void matrix::initialize_format(const matrix& source)
{
	initialize_format(source.width, source.height, source.depth);
}

void matrix::set_ptr_as_source(float* given_ptr)
{
	if (owning_data)
	{
		throw std::runtime_error("cannot set ptr as source if owning data");
	}
	if (given_ptr == nullptr)
	{
		throw std::runtime_error("given ptr is null");
	}
	data = given_ptr;
}

void matrix::set_all(float value)
{
	if_not_initialized_throw();

	for (int i = 0; i < item_count(); i++)
	{
		data[i] = value;
	}
}

void matrix::apply_noise(float range)
{
	if_not_initialized_throw();

	for (int i = 0; i < item_count(); i++)
	{
		data[i] += random_float_incl(-range, range);
	}
}

void matrix::mutate(float range)
{
	if_not_initialized_throw();

	add_at_flat(
		random_idx((int)item_count()),
		random_float_incl(-range, range));
}

size_t matrix::get_width() const
{
	return width;
}

size_t matrix::get_height() const
{
	return height;
}

size_t matrix::get_depth() const
{
	return depth;
}

size_t matrix::item_count() const
{
	return width * height * depth;
}


float matrix::get_at_flat(size_t idx) const
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	return data[idx];
}

void matrix::set_at_flat(size_t idx, float value)
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	data[idx] = value;
}

void matrix::add_at_flat(size_t idx, float value)
{
	if_not_initialized_throw();

	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	data[idx] += value;
}

float* matrix::get_data()
{
	return data;
}

const float* matrix::get_data_readonly() const
{
	return data;
}

//THESE ARE NOT TESTED
float* matrix::get_ptr_layer(size_t depth_idx)
{
	if_not_initialized_throw();
	return sub_ptr<float>(data, width * height, depth_idx);
}

float* matrix::get_ptr_row(size_t height_idx, size_t depth_idx)
{
	return get_ptr_layer(depth_idx) + height_idx * width;
}

float* matrix::get_ptr_item(size_t width_idx, size_t height_idx, size_t depth_idx)
{
	return get_ptr_row(height_idx, depth_idx) + width_idx;
}

void matrix::set_at(size_t x, size_t y, size_t z, float value)
{
	if_not_initialized_throw();
	data[get_idx(x, y, z)] = value;
}

void matrix::add_at(size_t x, size_t y, size_t z, float value)
{
	if_not_initialized_throw();
	data[get_idx(x, y, z)] += value;
}

void matrix::set_at(size_t x, size_t y, float value)
{
	set_at(x, y, 0, value);
}

void matrix::add_at(size_t x, size_t y, float value)
{
	add_at(x, y, 0, value);
}

float matrix::get_at(size_t x, size_t y, int z) const
{
	if_not_initialized_throw();
	return data[get_idx(x, y, z)];
}

float matrix::get_at(size_t x, size_t y) const
{
	return get_at(x, y, 0);
}

/*
const matrix& matrix::rotate180copy() const
{
	//NOT TESTED
	//ALSO - this is a very inefficient way to do this. I should be able to do this in-place

	matrix result(width, height, depth);
	for (int z = 0; z < depth; z++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				result.set_at(x, y, z, get_at(width - x - 1, height - y - 1, z));
			}
		}
	}
	return result;
}
*/

void matrix::dot_product(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (a.width != b.height || a.depth != b.depth)
	{
		throw std::invalid_argument("dot product could not be performed. input matrices are in the wrong format");
	}
	if (result.width != b.width || result.height != a.height || result.depth != a.depth)
	{
		throw std::invalid_argument("dot product could not be performed. result matrix is not the correct size");
	}

	for (int z = 0; z < result.depth; z++)
	{
		for (int y = 0; y < result.height; y++)
		{
			for (int x = 0; x < result.width; x++)
			{
				float sum = 0;
				for (int i = 0; i < a.width; i++)
				{
					sum += a.get_at(i, y, z) * b.get_at(x, i, z);
				}
				result.set_at(x, y, z, sum);
			}
		}
	}
}

void matrix::dot_product_flat(const matrix& a, const matrix& flat, matrix& result_flat)
{
	a.if_not_initialized_throw();
	flat.if_not_initialized_throw();
	result_flat.if_not_initialized_throw();

	if (a.width != flat.item_count() ||
		a.height != result_flat.item_count() ||
		a.depth != 1 ||
		result_flat.depth != 1)
	{
		throw std::invalid_argument("dot product could not be performed. input matrices are in the wrong format");
	}

	for (int x = 0; x < a.width; x++)
	{
		for (int y = 0; y < a.height; y++)
		{
			result_flat.add_at_flat(y, a.get_at(x, y) * flat.get_at_flat(x));
		}
	}
}

void matrix::add(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (a.width != b.width || a.height != b.height || a.depth != b.depth)
	{
		throw std::invalid_argument("addition could not be performed. input matrices are in the wrong format");
	}
	if (result.width != a.width || result.height != a.height || result.depth != a.depth)
	{
		throw std::invalid_argument("addition could not be performed. result matrix is not the correct size");
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		result.data[i] = a.data[i] + b.data[i];
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
		result.data[i] = a.data[i] + b.data[i];
	}
}

void matrix::subtract(const matrix& a, const matrix& b, matrix& result)
{
	a.if_not_initialized_throw();
	b.if_not_initialized_throw();
	result.if_not_initialized_throw();

	if (!equal_format(a, b) ||
		!equal_format(b, result) ||
		!equal_format(result, a))
	{
		throw std::invalid_argument("subtraction could not be performed. input matrices are in the wrong format");
	}

	for (int i = 0; i < a.item_count(); i++)
	{
		result.data[i] = a.data[i] - b.data[i];
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
		if (std::abs(a.data[i] - b.data[i]) > tolerance)
		{
			return false;
		}
	}
	return true;
}

bool matrix::equal_format(const matrix& a, const matrix& b)
{
	return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

void matrix::valid_cross_correlation(
	const matrix& input,
	const std::vector<matrix>& kernels,
	matrix& output,
	int stride)
{
	input.if_not_initialized_throw();
	output.if_not_initialized_throw();
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

	for (int z = 0; z < output.depth; z++)
	{
		for (int y = 0; y < output.height; y++)
		{
			for (int x = 0; x < output.width; x++)
			{
				float sum = 0;
				for (int curr_depth = 0; curr_depth < kernels[0].get_depth(); curr_depth++)
				{
					for (int i = 0; i < kernel_size; i++)
					{
						for (int j = 0; j < kernel_size; j++)
						{
							sum +=
								input.get_at(
									x * stride + i,
									y * stride + j,
									curr_depth) *
								kernels[z].get_at(
									i,
									j,
									curr_depth);
						}
					}
				}
				//if we do this, the output of all depths will be added together
				output.add_at(x, y, z, sum);
			}
		}
	}
}

//void matrix::valid_convolution(const matrix& input, const matrix& kernel, matrix& output)
//{
	//valid_cross_correlation(input, kernel.rotate180copy(), output);
//}

//void matrix::full_cross_correlation(const matrix& input, const matrix& kernel, matrix& output, int stride)
//{
	/*
	//this only works with a stride of one
	const size_t input_size = input.get_width();
	const size_t kernel_size = kernel.get_width();
	const size_t output_size = output.get_width();
	const size_t expected_output_size = ((input_size - kernel_size) / float(stride)) + 1; //(input_width - kernel_width) / (float)stride + 1
	if (output_size != expected_output_size)
	{
		throw std::invalid_argument("cross correlation could not be performed. output matrix is not the correct size");
	}
	if (input.get_depth() != kernel.get_depth() || input.get_depth() != output.get_depth())
	{
		throw std::invalid_argument("cross correlation could not be performed. input matrices are in the wrong format");
	}
	for (int z = 0; z < output.depth; z++)
	{
		for (int y = 0; y < output.height; y++)
		{
			for (int x = 0; x < output.width; x++)
			{
				float sum = 0;
				for (int i = 0; i < kernel_size; i++)
				{
					for (int j = 0; j < kernel_size; j++)
					{
						sum +=
							input.get_at(
								x + i * stride,
								y + j * stride,
								z) *
							kernel.get_at(i, j, z);
					}
				}
				output.set_at(x, y, z, sum);
			}
		}
	}
	*/
	//}

void matrix::scalar_multiplication(float a)
{
	if_not_initialized_throw();
	for (int i = 0; i < item_count(); i++)
	{
		data[i] *= a;
	}
}

void matrix::apply_activation_function(e_activation_t activation_fn)
{
	if_not_initialized_throw();
	for (int i = 0; i < item_count(); i++)
	{
		data[i] = ACTIVATION[activation_fn](data[i]);
	}
}

std::string matrix::get_string() const
{
	if_not_initialized_throw();

	std::string ret_val = "";

	for (int z = 0; z < depth; z++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				ret_val += std::to_string(get_at(x, y, z)) + " ";
			}
			ret_val += "\n";
		}
		ret_val += "\n";
	}

	return ret_val;
}