#include "test_util.hpp"

bool float_vectors_equal(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    const float max_diff = 0.000001f;

    for (size_t i = 0; i < vec1.size(); i++) {
		const float diff = std::abs(vec1[i] - vec2[i]);
        if (diff > max_diff) {
			return false;
		}
	}
}

std::wstring string_to_wstring(const std::string& str)
{
    return std::wstring(str.begin(), str.end());
}
