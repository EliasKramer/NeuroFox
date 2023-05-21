#pragma once
#include "data_space.hpp"
#include "neural_network.hpp"
class mnist_digit_overlord
{
private:
	data_space ds_training;
	data_space ds_test;
	neural_network nn;

	void label_to_matrix(unsigned char label, matrix& m) const;
	
	void load_data(
		data_space& ds, 
		std::string data_path, 
		std::string label_path);

	//returns the flat index of the float with the highest value in the matrix
	size_t idx_of_max(const matrix& m) const;
public:
	mnist_digit_overlord();

	void test();
	void train();
};