#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "Network.h"

TEST(NetworkTest, Single) {
	/******** layer 0 ********/
	float l0_weights_a[1]{
		0.5,
	};
	const quicknet::matrix_t l0_weights{1,1,l0_weights_a};

	float l0_bias_a[1]{
		0.0
	};
	const quicknet::vector_t l0_bias{1,l0_bias_a};

	float l0_output_a[1]{
		0.0
	};
	quicknet::vector_t l0_output{1,l0_output_a};

	/******** layers ********/
	quicknet::Layer layers[1]{
		{l0_weights, l0_bias, l0_output, nullptr}
	};

	/******** network ********/
	quicknet::Network network{1, layers};

	std::vector<float> inputs{1.0, 3.5, 0.25, -1.0};

	for(float in : inputs) {
		float input_a[1]{
			in
		};

		quicknet::vector_t input{1,input_a};
		const quicknet::vector_t& output = network.feedForward(input);

		ASSERT_EQ(in * 0.5, output(0));
	}
}

TEST(NetworkTest, Double) {
	/******** layer 0 ********/
	float l0_weights_a[2]{
		0.5,
		2.0
	};
	const quicknet::matrix_t l0_weights{2,1, l0_weights_a};

	float l0_bias_a[2]{
		-1.0, 1.0
	};
	const quicknet::vector_t l0_bias{2,l0_bias_a};

	float l0_output_a[2]{
		0.0, 0.0
	};
	quicknet::vector_t l0_output{2,l0_output_a};

	/******** layer 1 ********/
	float l1_weights_a[2]{
		2.0, 0.5
	};
	const quicknet::matrix_t l1_weights{1,2, l1_weights_a};

	float l1_bias_a[1]{
		1.0
	};
	const quicknet::vector_t l1_bias{1,l1_bias_a};

	float l1_output_a[1]{
		0.0
	};
	quicknet::vector_t l1_output{1,l1_output_a};

	/******** layers ********/
	quicknet::Layer layers[2]{
		{l0_weights, l0_bias, l0_output, nullptr},
		{l1_weights, l1_bias, l1_output, nullptr}
	};

	/******** network ********/
	quicknet::Network network{2, layers};

	std::vector<float> inputs{1.0, 3.5, 0.25, -1.0};

	for(float in : inputs) {
		float input_a[1]{
			in
		};

		quicknet::vector_t input{1,input_a};
		const quicknet::vector_t& output = network.feedForward(input);

		ASSERT_EQ(2.0 * in - 0.5, output(0));
	}
}
