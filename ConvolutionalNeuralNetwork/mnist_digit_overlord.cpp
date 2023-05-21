#include "mnist_digit_overlord.hpp"
#include <filesystem>
#include <fstream>

void mnist_digit_overlord::label_to_matrix(unsigned char label, matrix& m) const
{
	m = matrix(1, 10, 1);
	m.set_at(0, label, 1);
}

void mnist_digit_overlord::load_data(
	data_space& ds,
	std::string data_path,
	std::string label_path)
{
	std::vector<matrix> data_collection;
	std::vector<matrix> label_collection;

	std::filesystem::path path1 = std::filesystem::current_path();
	path1 = path1.lexically_normal() / data_path;
	std::filesystem::path path2 = std::filesystem::current_path();
	path2 = path2.lexically_normal() / label_path;

	std::string full_data_path = path1.string();
	std::string full_label_path = path2.string();

	std::cout << "reading images from " << full_data_path << std::endl;
	std::cout << "reading labels from " << full_label_path << std::endl;

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

	matrix current_image(28, 28, 1);

	for (int i = 0; i < num_images; i++) {

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {

				int pixel_idx = i * rows * cols + j * cols + k;
				//why is this "reading invalid data from image_buffer" ?
				unsigned char pixel = image_buffer[pixel_idx];

				current_image.set_at(j, k, (float)pixel / 255.0f);
			}
		}

		unsigned char label = label_buffer[i];
		matrix curr_label;
		label_to_matrix(label, curr_label);
		label_collection.push_back(curr_label);

		//push back creates a copy of the object
		data_collection.push_back(current_image);
	}

	delete[] image_buffer;
	delete[] label_buffer;

	data_file.close();
	label_file.close();

	ds = data_space(
		matrix(28, 28, 1),
		matrix(1, 10, 1),
		data_collection,
		label_collection);
}

size_t mnist_digit_overlord::idx_of_max(const matrix& m) const
{
	size_t idx = 0;
	float max = m.get_at(0, 0);
	for (size_t idx = 1; idx < m.item_count(); idx++)
	{
		float curr = m.get_at_flat(idx);
		if (curr > max)
		{
			max = curr;
			idx = idx;
		}
	}
	return idx;
}

mnist_digit_overlord::mnist_digit_overlord()
{
	std::string base_path = "..\\data\\digit_recognition";
	load_data(
		ds_training,
		base_path + "\\train-images.idx3-ubyte",
		base_path + "\\train-labels.idx1-ubyte");
	load_data(
		ds_test,
		base_path + "\\t10k-images.idx3-ubyte",
		base_path + "\\t10k-labels.idx1-ubyte");

	nn.set_input_format(matrix(28, 28, 1));
	nn.add_fully_connected_layer(16, e_activation_t::sigmoid_fn);
	nn.add_fully_connected_layer(16, e_activation_t::sigmoid_fn);
	nn.add_fully_connected_layer(matrix(1, 10, 1), e_activation_t::sigmoid_fn);
	nn.set_all_parameter(0);
}

void mnist_digit_overlord::test()
{
	ds_test.iterator_reset();
	size_t correct = 0;
	size_t total = 0;

	do
	{
		matrix input = ds_test.get_next_data();
		matrix label = ds_test.get_next_label();
		nn.forward_propagation_cpu(input);
		size_t idx = idx_of_max(nn.get_output());
		size_t label_idx = idx_of_max(label);
		if (idx == label_idx)
		{
			correct++;
		}
		ds_test.iterator_next();
		total++;
	} while (ds_test.iterator_has_next());
}

void mnist_digit_overlord::train()
{
}