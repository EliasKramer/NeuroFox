#pragma once

enum _activation {
	sigmoid_fn,
	relu_fn
} typedef activation;

float sigmoid(float x);
float relu(float x);

float sigmoid_derivative(float x);
float relu_derivative(float x);

//activation function pointer
using activation_fn = float(*)(float);

const activation_fn ACTIVATION[] =
{ sigmoid, relu };

const activation_fn DERIVATIVE[] =
{ sigmoid_derivative, relu_derivative };