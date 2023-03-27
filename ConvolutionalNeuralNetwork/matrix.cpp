#include "matrix.hpp"

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

void resize_matrix(matrix& m, int width, int height, int depth)
{
	m.width = width;
	m.height = height;
	m.depth = depth;
	m.data.resize(width * height * depth);
}

void set_all(matrix& m, float value)
{
	for (int i = 0; i < m.data.size(); i++)
	{
		m.data[i] = value;
	}
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
				ret_val += std::to_string(get_at(m, x, y, z)) + " ";
			}
			ret_val += "\n";
		}
		ret_val += "\n";
	}

	return ret_val;
}

void set_at(matrix& m, int x, int y, int z, float value)
{
	m.data[get_idx(m, x, y, z)] = value;
}

void set_at(matrix& m, int x, int y, int value)
{
	set_at(m, x, y, 0, value);
}

float get_at(const matrix& m, int x, int y, int z)
{
	return m.data[get_idx(m, x, y, z)];
}

float get_at(const matrix& m, int x, int z)
{
	return get_at(m, x, 0, z);
}

void matrix_dot(const matrix& a, const matrix& b, matrix& result)
{
	if (a.width != b.height || a.depth != b.depth)
	{
		throw "dot product could not be performed. input matrices are in the wrong format";
	}
	if (result.width != b.width || result.height != a.height || result.depth != a.depth)
	{
		throw "dot product could not be performed. result matrix is not the correct size";
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
					sum += get_at(a, i, y, z) * get_at(b, x, i, z);
				}
				set_at(result, x, y, z, sum);
			}
		}
	}
}

void matrix_add(const matrix& a, const matrix& b, matrix& result)
{
	if (a.width != b.width || a.height != b.height || a.depth != b.depth)
	{
		throw "addition could not be performed. input matrices are in the wrong format";
	}
	if (result.width != a.width || result.height != a.height || result.depth != a.depth)
	{
		throw "addition could not be performed. result matrix is not the correct size";
	}

	for (int i = 0; i < a.data.size(); i++)
	{
		result.data[i] = a.data[i] + b.data[i];
	}
}

void matrix_apply_activation(matrix& m, activation activation_fn)
{
	for (int i = 0; i < m.data.size(); i++)
	{
		m.data[i] = ACTIVATION[activation_fn](m.data[i]);
	}
}

bool are_equal(const matrix& a, const matrix& b)
{
	if (a.width != b.width || a.height != b.height || a.depth != b.depth)
	{
		return false;
	}

	for (int i = 0; i < a.data.size(); i++)
	{
		if (a.data[i] != b.data[i])
		{
			return false;
		}
	}

	return true;
}
