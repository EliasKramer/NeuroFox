#pragma once
#include "matrix.hpp"
#include "layer.hpp"

enum _pooling_type
{
	max_pooling,
	min_pooling,
	average_pooling
} typedef e_pooling_type_t;

class pooling_layer : public layer {
private:
	int filter_size;
	int stride;
	e_pooling_type_t pooling_fn;

	bool gpu_enabled = false;
public:
	//constructor
	pooling_layer(
		int filter_size,
		int stride,
		e_pooling_type_t pooling_fn
	);
	pooling_layer(const pooling_layer& other);

	//clone
	std::unique_ptr<layer> clone() const override;

	size_t get_parameter_count() const override;

	void set_input_format(vector3 input_format) override;

	int get_filter_size() const;
	int get_stride() const;
	e_pooling_type_t get_pooling_fn() const;

	void set_all_parameters(float value) override;
	void apply_noise(float range) override;
	void mutate(float range) override;

	void sync_device_and_host() override;

	void forward_propagation(const matrix& input) override;
	void back_propagation(const matrix& input, matrix* passing_error) override;

	void apply_deltas(size_t training_data_count, float learning_rate) override;
	
	void enable_gpu_mode() override;
	void disable_gpu() override;

	bool equal_format(const layer& other) override;
	bool equal_parameter(const layer& other) override;
};