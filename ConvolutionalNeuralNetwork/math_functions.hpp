#pragma once

enum _activation {
	sigmoid,
	relu
} typedef activation;

float sigmoid_fn(float x);
float relu_fn(float x);


//activation function pointer
using activation_fn = float(*)(float);

const activation_fn ACTIVATION[] = { sigmoid_fn, relu_fn };
