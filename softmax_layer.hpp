#pragma once
#include "layer.hpp"

class softmax_layer: public layer {
public:
	softmax_layer();
	softmax_layer(std::ifstream& file);
	softmax_layer(const softmax_layer& other);
	~softmax_layer() override = default;
	std::unique_ptr<layer> clone() const override;
	bool is_parameter_layer() const override;
	size_t get_parameter_count() const override;
	void set_input_format(vector3 input_format) override;
	void set_error_for_last_layer(const matrix& expected) override;
	void set_all_parameters(float value) override;
	void apply_noise(float range) override;
	void mutate(float range) override;
	std::string parameter_analysis() const override;
	void forward_propagation(const matrix& input) override;
	void partial_forward_prop(const matrix& input, const matrix& prev_input, const vector3& change_idx) override;
	void partial_forward_prop(const matrix& input, float value, const vector3& change_idx) override;
	void back_propagation(const matrix& input, matrix* passing_error) override;
	void apply_deltas(size_t training_data_count, float learning_rate) override;
	void enable_gpu_mode() override;
	void disable_gpu() override;
	bool equal_format(const layer& other) override;
	bool equal_parameter(const layer& other) override;
	void set_parameters(const layer& other) override;
	void write_to_ofstream(std::ofstream& file) const override;
};