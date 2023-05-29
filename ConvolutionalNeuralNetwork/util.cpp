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

std::string byte_size_to_str(size_t byte_size)
{
	size_t giga = 1024 * 1024 * 1024;
	size_t mega = 1024 * 1024;
	size_t kilo = 1024;

	size_t giga_count = byte_size / giga;
	size_t mega_count = (byte_size - giga_count * giga) / mega;
	size_t kilo_count = (byte_size - giga_count * giga - mega_count * mega) / kilo;
	size_t byte_count = byte_size - giga_count * giga - mega_count * mega - kilo_count * kilo;

	std::string result = "";
	if (giga_count > 0)
		result += std::to_string(giga_count) + "GB ";
	if (mega_count > 0)
		result += std::to_string(mega_count) + "MB ";
	if (kilo_count > 0)
		result += std::to_string(kilo_count) + "KB ";
	if (byte_count > 0)
		result += std::to_string(byte_count) + "B ";

	return result == "" ? "0 B " : result;
}
