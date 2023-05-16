#include "matrix.hpp"
#include <numeric>

size_t matrix::get_idx(size_t x, size_t y, size_t z) const
{
	return x + y * width + z * width * height;
}

void matrix::check_for_valid_format() const
{
	if (width == 0 || height == 0 || depth == 0)
	{
		throw std::invalid_argument(
			"width, height and depth must be >=1");
	}
}

void matrix::allocate_mem()
{
	if (!owning_data)
	{
		throw std::runtime_error("cannot allocate if not owning");
	}
	if (data != nullptr)
	{
		throw std::runtime_error("cannot allocate if data will be overwritten");
	}
	data = new float[item_count()];
	set_all(0);
}

matrix::matrix(
) :
	width(0),
	height(0),
	depth(0),
	owning_data(false),
	data(nullptr)
{}

void matrix::copy_data(const float* src)
{
	if (src == nullptr)
		throw std::runtime_error("src is null");
	if (!owning_data)
		throw std::runtime_error("cannot copy data if it is now owned");
	if (data == nullptr)
		throw std::runtime_error("data is null");

	std::copy(src, src + item_count(), this->data);
}

void matrix::copy_data(const matrix& src)
{
	if (src.item_count() != item_count())
		throw std::runtime_error("cannot copy data if not the same size");
	copy_data(src.data);
}

void matrix::delete_data()
{
	if (owning_data && data != nullptr)
	{
		delete[] data;
		data = nullptr;
	}
}

matrix::matrix(
	size_t width,
	size_t height,
	size_t depth
) :
	width(width),
	height(height),
	depth(depth),
	owning_data(true),
	data(nullptr)
{
	check_for_valid_format();
	allocate_mem();
}

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
	if (copy)
	{
		allocate_mem();
		copy_data(given_ptr);
	}
}

matrix::matrix(
	size_t width,
	size_t height,
	size_t depth,
	const std::vector<float>& given_vector
) :
	width(width),
	height(height),
	depth(depth),
	owning_data(true),
	data(nullptr)
{
	check_for_valid_format();
	allocate_mem();
	copy_data(given_vector.data());
}

matrix::~matrix()
{
	delete_data();
}

std::unique_ptr<matrix> matrix::clone() const
{
	return std::make_unique<matrix>(width, height, depth, data, true);
}

void matrix::resize(size_t width, size_t height, size_t depth)
{
	this->width = width;
	this->height = height;
	this->depth = depth;
	check_for_valid_format();

	if (!owning_data && data == nullptr)
	{
		owning_data = true;
		allocate_mem();
	}
	else
	{
		throw std::runtime_error("resizing can only be done once after default constructor");
	}
}

void matrix::resize(const matrix& source)
{
	resize(source.width, source.height, source.depth);
}

matrix::matrix(const matrix& source)
	:matrix()
{
	resize(source);
	copy_data(source.data);
}

void matrix::set_all(float value)
{
	for (int i = 0; i < item_count(); i++)
	{
		data[i] = value;
	}
}

void matrix::apply_noise(float range)
{
	for (int i = 0; i < item_count(); i++)
	{
		data[i] += random_float_incl(-range, range);
	}
}

void matrix::mutate(float range)
{
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
	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	return data[idx];
}

void matrix::set_at_flat(size_t idx, float value)
{
	if (idx >= item_count())
	{
		throw std::invalid_argument("idx must be in range");
	}
	data[idx] = value;
}

void matrix::add_at_flat(size_t idx, float value)
{
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

void matrix::set_at(size_t x, size_t y, size_t z, float value)
{
	data[get_idx(x, y, z)] = value;
}

void matrix::add_at(size_t x, size_t y, size_t z, float value)
{
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
	for (int i = 0; i < item_count(); i++)
	{
		data[i] *= a;
	}
}

void matrix::apply_activation_function(e_activation_t activation_fn)
{
	for (int i = 0; i < item_count(); i++)
	{
		data[i] = ACTIVATION[activation_fn](data[i]);
	}
}

std::string matrix::get_string() const
{
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