#pragma once
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <random>

// a random number between 0 and size-1
int random_idx(int size);

//returns true with a probability of true_bias, false with a probability of false_bias
bool biased_coin_toss(float true_bias, float false_bias);

//a random number between min and max (inclusive)
float random_float_incl(float min, float max);