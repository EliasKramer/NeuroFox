#pragma once
#include "matrix.hpp"
#include "layer.hpp"

enum _pooling_type
{
	max_pooling,
	min_pooling,
	average_pooling
} typedef pooling_type;

class pooling_layer : public layer {
private:
	int filter_size;
	int stride;
	pooling_type pooling_fn;
public:
	//constructor
	pooling_layer(
		matrix* input,
		int filter_size,
		int stride,
		pooling_type pooling_fn
	);

	void forward_propagation() override;
	void back_propagation() override;
};