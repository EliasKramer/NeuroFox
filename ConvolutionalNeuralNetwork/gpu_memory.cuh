#pragma once
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <memory>
#include "matrix.hpp"

template<typename T>
class gpu_memory {
private:
    T* gpu_ptr;
    size_t element_count;

public:
    explicit gpu_memory(size_t item_count)
        : element_count(item_count) 
    {
        cudaError_t err = cudaMalloc(&gpu_ptr, byte_size());
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    void set_all(T value)
    {
        std::vector<T> init_values(element_count, value);

        cudaError_t err =
            cudaMemcpy(
                gpu_ptr,
                init_values.data(),
                byte_size(),
                cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize device memory");
        }
    }

    explicit gpu_memory(size_t item_count, T init_value)
        : gpu_memory(item_count)
    {
        set_all(init_value);
    }
    
    explicit gpu_memory(const matrix& m)
        : gpu_memory(m.flat_readonly().size())
    {
        //check if T is float
        if (sizeof(T) != sizeof(float))
        {
			throw std::runtime_error("gpu_memory only supports float");
		}

        if (m.flat_readonly().size() != element_count)
        {
            throw std::runtime_error("gpu_memory size mismatch");
        }

        cudaError_t err =
            cudaMemcpy(
				gpu_ptr,
				m.flat_readonly().data(),
				byte_size(),
				cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
			throw std::runtime_error("Failed to initialize device memory");
		}
    }

    ~gpu_memory()
    {
        cudaError_t err = cudaFree(gpu_ptr);
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to free device memory");
    }

    T* gpu_data_ptr() const 
    {
        return gpu_ptr;
    }

    std::unique_ptr<std::vector<T>> to_cpu() const
    {
        std::unique_ptr<std::vector<T>> cpu_data = 
            std::make_unique<std::vector<T>>(element_count);
        
        cudaError_t err =
            cudaMemcpy(
                cpu_data->data(),
                gpu_ptr,
                byte_size(),
                cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy device memory to host");
        }

        return std::move(cpu_data);
	}

    size_t byte_size() const 
    {
        return element_count * sizeof(T);
    }

    size_t item_count() const 
    {
		return element_count;
	}
};