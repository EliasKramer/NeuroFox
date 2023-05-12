#pragma once
#include "matrix.hpp"
#include "gpu_memory.cuh"

class nn_data
{
protected:
	matrix data;
	matrix label;
public:
	nn_data();
	nn_data(const matrix& data, const matrix& label);

	const matrix& get_data() const;
	const matrix& get_label() const;

	const matrix* get_data_p() const;
	const matrix* get_label_p() const;

	void set_data(const matrix& data);
	void set_label(const matrix& label);
};