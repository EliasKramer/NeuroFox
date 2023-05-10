#pragma once
#include <cstdlib>
#include <ctime>
#include <random>

// a random number between 0 and size-1
int random_idx(int size);

//returns true with a probability of true_bias, false with a probability of false_bias
bool biased_coin_toss(float true_bias, float false_bias);

//a random number between min and max (inclusive)
float random_float_incl(float min, float max);

bool is_whole_number(float number);

/// <summary>
/// splits the pointer into multiple chunks
/// these chunks are the size of elements * sizeof(T)
/// and you get the index'th pointer
/// </summary>
/// <param name="ptr">the ptr that will be split</param>
/// <param name="elements">the amount of elements that each chunk has</param>
/// <param name="index">the index of the chunk</param>
/// <returns>pointer to the index'th chunk that got split</returns>
template<typename T>
T* sub_ptr(T* ptr, size_t elements, size_t index)
{
	//return (T*)(((char*)ptr) + index * elements * sizeof(T))
	return ptr + index * elements;
}