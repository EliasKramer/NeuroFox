#include "vector3.hpp"

vector3::vector3()
:x(0), y(0), z(0)
{}

vector3::vector3(size_t x, size_t y, size_t z)
	:x(x), y(y), z(z)
{}

vector3::vector3(size_t x, size_t y)
	:x(x), y(y), z(0)
{}

vector3::vector3(size_t x)
	:x(x), y(0), z(0)
{}

bool vector3::is_in_bounds(const vector3 & position) const
{
	return position.x < x && position.y < y && position.z < z;
}

size_t vector3::get_index(const vector3& format) const
{
	if(!format.is_in_bounds(*this))
		throw std::invalid_argument("vector3::get_index: format is not in bounds");

	//x + y * width + z * width * height
	return x + y * format.x + z * format.x * format.y;
}

size_t vector3::item_count() const
{
	return x * y * z;
}

bool vector3::are_equal(const vector3 & v1, const vector3 & v2)
{
	return 
		v1.x == v2.x &&
		v1.y == v2.y &&
		v1.z == v2.z;
}