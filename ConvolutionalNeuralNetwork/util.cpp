#include "util.hpp"

int random_idx(int size)
{
    // Seed the random number generator with the current time
    std::srand((int)std::time(nullptr));

    // Generate a random number between 0 and size-1
    return std::rand() % size;
}

bool biased_coin_toss(float true_bias, float false_bias)
{
    if(true_bias <= 0 || false_bias <= 0)
		throw std::invalid_argument("true_bias and false_bias must be greater than 0");

    // Seed the random number generator with the current time
	std::srand((int)std::time(nullptr));
	// Generate a random number between 0 and 1
	float random_number = (float)std::rand() / (float)RAND_MAX;
	// Return true if the random number is less than the bias, false otherwise
	return random_number < true_bias / (true_bias + false_bias);
}

float random_float_incl(float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, std::nextafter(max, std::numeric_limits<float>::max()));
    return dis(gen);
}

bool is_whole_number(float number)
{
    return number == (int)number;
}

int swap_endian(int value) {
	int result = 0;
	result |= (value & 0xFF) << 24;
	result |= ((value >> 8) & 0xFF) << 16;
	result |= ((value >> 16) & 0xFF) << 8;
	result |= ((value >> 24) & 0xFF);
	return result;
}
//determins wether the system is little endian or big endian
bool is_little_endian()
{
	int num = 1;
	return (*(char*)&num == 1);
}