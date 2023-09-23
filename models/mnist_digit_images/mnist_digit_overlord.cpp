#include "mnist_digit_overlord.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

void mnist_digit_overlord::label_to_matrix(unsigned char label, matrix& m) const
{
	m = matrix(vector3(1, 10, 1));
	m.set_at_host(vector3(0, label), 1);
}

float mnist_digit_overlord::get_digit_cost(const matrix& output, const matrix& label) const
{
	float cost = 0;
	for (size_t i = 0; i < output.item_count(); i++)
	{
		cost +=
			(output.get_at_flat_host(i) - label.get_at_flat_host(i)) *
			(output.get_at_flat_host(i) - label.get_at_flat_host(i));
	}
	return cost;
}

void mnist_digit_overlord::print_digit_image(const matrix& m) const
{
	for (int x = 0; x < m.get_width(); x++)
	{
		for (int y = 0; y < m.get_height(); y++)
		{
			m.get_at_host(vector3(x, y)) > 0.5 ?
				std::cout << "# " :
				std::cout << ". ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void mnist_digit_overlord::load_data(
	data_space& ds,
	std::string data_path,
	std::string label_path)
{
	std::filesystem::path path1 = std::filesystem::current_path();
	path1 = path1.lexically_normal() / data_path;
	std::filesystem::path path2 = std::filesystem::current_path();
	path2 = path2.lexically_normal() / label_path;

	std::string full_data_path = path1.string();
	std::string full_label_path = path2.string();

	//std::cout << "reading images from " << full_data_path << std::endl;
	//std::cout << "reading labels from " << full_label_path << std::endl;

	//check if files exists
	if (!std::filesystem::exists(std::filesystem::path(full_data_path)) ||
		!std::filesystem::exists(std::filesystem::path(full_label_path)))
	{
		throw std::runtime_error("A file does not exist");
	}

	//Open the data file and read the magic number and number of images
	//The magic number is there to check if the file is read correctly
	std::ifstream data_file(full_data_path, std::ios::binary);
	int magic_number, num_images, rows, cols;
	//can be improved with structs
	data_file.read((char*)&magic_number, sizeof(magic_number));
	data_file.read((char*)&num_images, sizeof(num_images));
	data_file.read((char*)&rows, sizeof(rows));
	data_file.read((char*)&cols, sizeof(cols));

	//the magic number is stored in big endian,
	//so we need to swap the bytes,
	//if we are on a little endian system
	if (is_little_endian())
	{
		magic_number = swap_endian(magic_number);
		num_images = swap_endian(num_images);
		rows = swap_endian(rows);
		cols = swap_endian(cols);
	}

	// Open the label file and read the magic number and number of labels
	std::ifstream label_file(full_label_path, std::ios::binary);
	int label_magic_number, num_labels;
	label_file.read((char*)&label_magic_number, sizeof(label_magic_number));
	label_file.read((char*)&num_labels, sizeof(num_labels));

	if (is_little_endian())
	{
		label_magic_number = swap_endian(label_magic_number);
		num_labels = swap_endian(num_labels);
	}

	// Check that the magic numbers and number of items match
	if (magic_number != 2051 || label_magic_number != 2049 || num_images != num_labels) {
		throw std::runtime_error("Invalid MNIST data files");
	}

	//read all pixel values and labels at once,
	//because reading from a file is a very costly operation
	int image_buffer_size = num_images * rows * cols;
	char* image_buffer = new char[image_buffer_size];
	data_file.read(image_buffer, image_buffer_size);

	char* label_buffer = new char[num_labels];
	label_file.read(label_buffer, num_labels);


	vector3 image_format(28, 28, 1);
	vector3 label_format(1, 10, 1);

	matrix current_image(image_format);
	matrix curr_label(label_format);

	ds = data_space(num_images, image_format, label_format);

	for (int i = 0; i < num_images; i++) {

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {

				int pixel_idx = i * rows * cols + j * cols + k;
				//why is this "reading invalid data from image_buffer" ?
				unsigned char pixel = image_buffer[pixel_idx];

				current_image.set_at_host(vector3(j, k), (float)pixel / 255.0f);
			}
		}

		unsigned char label = label_buffer[i];
		label_to_matrix(label, curr_label);
		ds.set_label(curr_label, i);
		ds.set_data(current_image, i);
	}

	delete[] image_buffer;
	delete[] label_buffer;

	data_file.close();
	label_file.close();
}

size_t mnist_digit_overlord::idx_of_max(const matrix& m) const
{
	size_t max_idx = 0;
	float max = m.get_at_host(vector3(0, 0));
	for (size_t idx = 1; idx < m.item_count(); idx++)
	{
		float curr = m.get_at_flat_host(idx);
		if (curr > max)
		{
			max = curr;
			max_idx = idx;
		}
	}
	return max_idx;
}

void mnist_digit_overlord::enable_gpu()
{
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "copying to gpu" << std::endl;
	ds_training.copy_to_gpu();
	ds_test.copy_to_gpu();
	nn.enable_gpu_mode();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout
		<< "copied to gpu, took "
		<< ms_to_str(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())
		<< std::endl;
}

mnist_digit_overlord::mnist_digit_overlord()
{
	std::string base_path = "..\\models\\mnist_digit_images\\data\\digit_recognition";
	std::cout << "loading training data" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	load_data(
		ds_training,
		base_path + "\\train-images.idx3-ubyte",
		base_path + "\\train-labels.idx1-ubyte");
	std::cout << "loading test data" << std::endl;

	load_data(
		ds_test,
		base_path + "\\t10k-images.idx3-ubyte",
		base_path + "\\t10k-labels.idx1-ubyte");
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "data loaded, took " <<
		ms_to_str(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) <<
		std::endl;

	nn.set_input_format(vector3(28, 28, 1));
	//nn.add_pooling_layer(2, 2, e_pooling_type_t::average_pooling);
	//nn.add_convolutional_layer(2, 3, 1, e_activation_t::sigmoid_fn);
	//nn.add_fully_connected_layer(16, e_activation_t::sigmoid_fn);
	nn.add_fully_connected_layer(26, e_activation_t::leaky_relu_fn);
	nn.add_fully_connected_layer(26, e_activation_t::leaky_relu_fn);
	nn.add_fully_connected_layer(vector3(1, 10, 1), e_activation_t::identity_fn);
	nn.add_softmax_layer();

	//nn.apply_noise(.1);
	nn.xavier_initialization();
	//enable_gpu();
}

void mnist_digit_overlord::debug_function()
{
	neural_network before = nn;
	save_to_file();
	load_from_file();
	neural_network after = nn;

	bool same_format = before.nn_equal_format(after);
	bool same_parameters = before.equal_parameter(after);

	std::cout << "same format: " << same_format << std::endl;
	std::cout << "same parameters: " << same_parameters << std::endl;
}

void mnist_digit_overlord::save_to_file()
{
	std::string path = "..\\models\\mnist_digit_images\\saved\\digit_model.parameter";

	std::cout << "saving to file " << path << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	nn.save_to_file(path);

	auto end = std::chrono::high_resolution_clock::now();
	std::cout
		<< "saved to file, took "
		<< ms_to_str(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())
		<< std::endl;
}

void mnist_digit_overlord::load_from_file()
{
	std::string path = "..\\models\\mnist_digit_images\\saved\\digit_model.parameter";
	std::cout << "loading from file " << path << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	nn = neural_network(path);

	auto end = std::chrono::high_resolution_clock::now();
	std::cout
		<< "loaded from file, took "
		<< ms_to_str(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())
		<< std::endl;
}

void mnist_digit_overlord::print_nn_size() const
{
	std::cout << "nn size: " << nn.get_param_count() << std::endl;
}

test_result mnist_digit_overlord::test()
{
	smart_assert(ds_test.is_in_gpu_mode() == nn.is_in_gpu_mode());
	test_result result;
	
	size_t correct = 0;
	float cost_sum = 0;
	size_t total = 0;

	auto start = std::chrono::high_resolution_clock::now();

	matrix input(vector3(28, 28, 1));
	matrix label(vector3(1, 10, 1));
	if (ds_test.is_in_gpu_mode())
	{
		input.enable_gpu_mode();
		label.enable_gpu_mode();
	}

	for (int i = 0; i < ds_test.get_item_count(); i++)
	{
		ds_test.observe_data_at_idx(input, i);
		ds_test.observe_label_at_idx(label, i);
		nn.forward_propagation(input);
		nn.get_output().sync_device_and_host();

		cost_sum += get_digit_cost(nn.get_output_readonly(), label);

		size_t idx = idx_of_max(nn.get_output_readonly());
		size_t label_idx = idx_of_max(label);
		if (idx == label_idx)
		{
			correct++;
		}
		total++;
	}

	auto end = std::chrono::high_resolution_clock::now();

	result.accuracy = (float)correct / (float)total;
	result.data_count = total;
	result.time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	result.avg_cost = (float)cost_sum / (float)total;

	return result;
}

test_result mnist_digit_overlord::test_on_training_data()
{
	smart_assert(ds_test.is_in_gpu_mode() == nn.is_in_gpu_mode());
	test_result result;

	size_t correct = 0;
	float cost_sum = 0;
	size_t total = 0;

	auto start = std::chrono::high_resolution_clock::now();

	matrix input(vector3(28, 28, 1));
	matrix label(vector3(1, 10, 1));
	if (ds_training.is_in_gpu_mode())
	{
		input.enable_gpu_mode();
		label.enable_gpu_mode();
	}

	for (int i = 0; i < ds_training.get_item_count(); i++)
	{
		ds_training.observe_data_at_idx(input, i);
		ds_training.observe_label_at_idx(label, i);
		nn.forward_propagation(input);
		nn.get_output().sync_device_and_host();

		cost_sum += get_digit_cost(nn.get_output_readonly(), label);

		size_t idx = idx_of_max(nn.get_output_readonly());
		size_t label_idx = idx_of_max(label);
		if (idx == label_idx)
		{
			correct++;
		}
		total++;
	}

	auto end = std::chrono::high_resolution_clock::now();

	result.accuracy = (float)correct / (float)total;
	result.data_count = total;
	result.time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	result.avg_cost = (float)cost_sum / (float)total;

	return result;
}

void mnist_digit_overlord::train(
	size_t epochs,
	size_t batch_size,
	float learning_rate)
{
	std::cout
		<< "start training... "
		<< "(epochs:" << epochs
		<< ", batch_size:" << batch_size
		<< ", learning_rate:" << learning_rate
		<< ")"
		<< std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	nn.learn_on_ds(ds_training, epochs, batch_size, learning_rate, false);
	/*
	float inital_learning_rate = learning_rate;
	float current_learing_rate = inital_learning_rate;
	float percent_correct = 0;
	for (size_t i = 0; i < epochs; i++)
	{
		nn.learn_on_ds(ds_training, 1, batch_size, current_learing_rate, false);
		test_result res = test();
		float perc_diff = res.accuracy - percent_correct;
		std::cout << res.to_string();
		std::cout << "epoch " << i << ": " << res.accuracy << " (" << perc_diff << ")" << std::endl;
		//current_learing_rate = 0.9f * current_learing_rate;
		percent_correct = res.accuracy;
		std::cout << "learning rate: " << current_learing_rate << std::endl;
	}
	*/
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "training finished, took " <<
		ms_to_str(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) <<
		std::endl;
}