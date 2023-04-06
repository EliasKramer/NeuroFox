#include "matrix.hpp"
#include <numeric>

static int get_idx(const matrix& m, int x, int y, int z)
{
	return x + y * m.width + z * m.width * m.height;
}

matrix* create_matrix(int width, int height, int depth)
{
	matrix* m = new matrix;
	resize_matrix(*m, width, height, depth);

	return m;
}

matrix get_matrix(int width, int height, int depth)
{
	matrix m;
	resize_matrix(m, width, height, depth);
	return m;
}

size_t matrix_hash(const matrix& m)
{
	return std::accumulate(m.data.begin(), m.data.end(), 0,
		[](size_t h, float f) {
			return h + std::hash<float>()(f);
		});
}

void resize_matrix(matrix& m, int width, int height, int depth)
{
	m.width = width;
	m.height = height;
	m.depth = depth;
	m.data.resize(width * height * depth);
}

void resize_matrix(matrix& resizing_matrix, const matrix& source)
{
	resize_matrix(resizing_matrix, source.width, source.height, source.depth);
}

bool matrix_equal_format(const matrix& a, const matrix& b)
{
	return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

void set_all(matrix& m, float value)
{
	for (int i = 0; i < m.data.size(); i++)
	{
		m.data[i] = value;
	}
}

void matrix_apply_noise(matrix& m, float range)
{
	for (int i = 0; i < m.data.size(); i++)
	{
		m.data[i] += random_float_incl(-range, range);
	}
}

void matrix_mutate(matrix& m, float range)
{
	m.data[random_idx(m.data.size())] += random_float_incl(-range, range);
}

std::vector<float>& matrix_flat(matrix& m)
{
	return m.data;
}

const std::vector<float>& matrix_flat_readonly(const matrix& m)
{
	return m.data;
}

void set_at(matrix& m, int x, int y, int z, float value)
{
	m.data[get_idx(m, x, y, z)] = value;
}

void set_at(matrix& m, int x, int y, int value)
{
	set_at(m, x, y, 0, value);
}

float matrix_get_at(const matrix& m, int x, int y, int z)
{
	return m.data[get_idx(m, x, y, z)];
}

float matrix_get_at(const matrix& m, int x, int y)
{
	return matrix_get_at(m, x, y, 0);
}

void matrix_dot(const matrix& a, const matrix& b, matrix& result)
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
					sum += matrix_get_at(a, i, y, z) * matrix_get_at(b, x, i, z);
				}
				set_at(result, x, y, z, sum);
			}
		}
	}
}

void matrix_dot_flat(const matrix& a, const matrix& flat, matrix& result_flat)
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
			result_flat.data[y] += matrix_get_at(a, x, y) * flat.data[x];
		}
	}
}

void matrix_add(const matrix& a, const matrix& b, matrix& result)
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

void matrix_add_flat(const matrix& a, const matrix& b, matrix& result)
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

void matrix_apply_activation(matrix& m, e_activation_t activation_fn)
{
	for (int i = 0; i < m.data.size(); i++)
	{
		m.data[i] = ACTIVATION[activation_fn](m.data[i]);
	}
}

bool are_equal(const matrix& a, const matrix& b)
{
	return matrix_hash(a) == matrix_hash(b);
}

std::string get_matrix_string(matrix& m)
{
	std::string ret_val = "";

	for (int z = 0; z < m.depth; z++)
	{
		for (int y = 0; y < m.height; y++)
		{
			for (int x = 0; x < m.width; x++)
			{
				ret_val += std::to_string(matrix_get_at(m, x, y, z)) + " ";
			}
			ret_val += "\n";
		}
		ret_val += "\n";
	}

	return ret_val;
}