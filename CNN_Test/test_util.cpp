#include "test_util.hpp"

bool are_float_vectors_equal(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    return std::equal(vec1.begin(), vec1.end(), vec2.begin());
}