#include "matrix.hpp"
#include <numeric>

int matrix::get_idx(int x, int y, int z) const
{
	return x + y * width + z * width * height;
}

matrix::matrix()
	:matrix(0, 0, 0)
{}

matrix::matrix(int width, int height, int depth)
{
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->data = std::vector<float>(width * height * depth);
}

size_t matrix::get_hash() const
{
	return std::accumulate(data.begin(), data.end(), 0,
		[](size_t h, float f) {
			return h + std::hash<float>()(f);
		});
}

void matrix::resize(int width, int height, int depth)
{
	this->width = width;
	this->height = height;
	this->depth = depth;

	data.resize(width * height * depth);
}

void matrix::resize(const matrix& source)
{
	resize(source.width, source.height, source.depth);
}

void matrix::set_all(float value)
{
	for (int i = 0; i < data.size(); i++)
	{
		data[i] = value;
	}
}

void matrix::apply_noise(float range)
{
	for (int i = 0; i < data.size(); i++)
	{
		data[i] += random_float_incl(-range, range);
	}
}

void matrix::mutate(float range)
{
	data[random_idx(data.size())] += random_float_incl(-range, range);
}

int matrix::get_width() const
{
	return width;
}

int matrix::get_height() const
{
	return height;
}

int matrix::get_depth() const
{
	return depth;
}

std::vector<float>& matrix::flat()
{
	return data;
}

const std::vector<float>& matrix::flat_readonly() const
{
	return data;
}

void matrix::set_at(int x, int y, int z, float value)
{
	data[get_idx(x, y, z)] = value;
}

void matrix::add_at(int x, int y, int z, float value)
{
	data[get_idx(x, y, z)] += value;
}

void matrix::set_at(int x, int y, float value)
{
	set_at(x, y, 0, value);
}

void matrix::add_at(int x, int y, float value)
{
	add_at(x, y, 0, value);
}

float matrix::get_at(int x, int y, int z) const
{
	return data[get_idx(x, y, z)];
}

float matrix::get_at(int x, int y) const
{
	return get_at(x, y, 0);
}

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
	if (a.width != flat.data.size() ||
		a.height != result_flat.data.size() ||
		a.depth != 1 ||
		result_flat.depth != 1)
	{
		throw std::invalid_argument("dot product could not be performed. input matrices are in the wrong format");
	}

	for (int x = 0; x < a.width; x++)
	{
		for (int y = 0; y < a.height; y++)
		{
			result_flat.data[y] += a.get_at(x, y) * flat.data[x];
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

	for (int i = 0; i < a.data.size(); i++)
	{
		result.data[i] = a.data[i] + b.data[i];
	}
}

void matrix::add_flat(const matrix& a, const matrix& b, matrix& result)
{
	if (a.data.size() != b.data.size())
	{
		throw std::invalid_argument("addition could not be performed. input matrices are not the same size");
	}

	for (int i = 0; i < a.data.size(); i++)
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

	for (int i = 0; i < a.data.size(); i++)
	{
		result.data[i] = a.data[i] - b.data[i];
	}
}

bool matrix::are_equal(const matrix& a, const matrix& b)
{
	return a.get_hash() == b.get_hash();
}

bool matrix::equal_format(const matrix& a, const matrix& b)
{
	return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

void matrix::valid_cross_correlation(const matrix& input, const matrix& kernel, matrix& output)
{
	//this only works with a stride of one
	const size_t input_size = input.get_width();
	const size_t kernel_size = kernel.get_width();
	const size_t output_size = output.get_width();
	const size_t expected_output_size = input_size - kernel_size + 1;
	
	if (output_size != expected_output_size)
	{
		throw std::invalid_argument("cross correlation could not be performed. output matrix is not the correct size");
	}
	if (input.get_depth() != kernel.get_depth())
	{
		throw std::invalid_argument("cross correlation could not be performed. input matrices are in the wrong format");
	}
	if (output.get_depth() != 1)
	{
		throw std::invalid_argument("cross correlation could not be performed. output matrix is in the wrong format");
	}

	output.set_all(0);

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
						sum += input.get_at(x + i, y + j, z) * kernel.get_at(i, j, z);
					}
				}
				//if we do this, the output of all depths will be added together
				output.add_at(x, y, z, sum);
			}
		}
	}
}

void matrix::valid_convolution(const matrix& input, const matrix& kernel, matrix& output)
{
	valid_cross_correlation(input, kernel.rotate180copy(), output);
}

void matrix::full_cross_correlation(const matrix& input, const matrix& kernel, matrix& output)
{
	//this only works with a stride of one
	const size_t input_size = input.get_width();
	const size_t kernel_size = kernel.get_width();
	const size_t output_size = output.get_width();
	const size_t expected_output_size = input_size + kernel_size - 1;
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
						if (x - i >= 0 && x - i < input_size && y - j >= 0 && y - j < input_size)
						{
							sum += input.get_at(x - i, y - j, z) * kernel.get_at(i, j, z);
						}
					}
				}
				output.set_at(x, y, z, sum);
			}
		}
	}
}

void matrix::scalar_multiplication(float a)
{
	for (int i = 0; i < data.size(); i++)
	{
		data[i] *= a;
	}
}

void matrix::apply_activation_function(e_activation_t activation_fn)
{
	for (int i = 0; i < data.size(); i++)
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