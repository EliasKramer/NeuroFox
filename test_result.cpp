#include "test_result.hpp"

std::string test_result::to_string()
{
	std::string result = "";
	result += "Data count: " + std::to_string(data_count) + "\n";
	result += "Time taken: " + ms_to_str(time_in_ms) + "\n";
	result += "data/ms: " + std::to_string((float)data_count / (float)time_in_ms) + "\n";
	result += "Avg cost: " + std::to_string(avg_cost) + "\n";
	result += "Avg cost sqrt: " + std::to_string(std::sqrt(avg_cost)) + "\n";
	result += "Accuracy: " + std::to_string(accuracy * 100) + "%\n";
	result += "Avg Diff: " + std::to_string(avg_diff) + "\n";
	return result;
}
