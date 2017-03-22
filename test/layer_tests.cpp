#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "Layer.h"

TEST(LayerTest, FeedForwardSimple) {
	float weights_a[1]{
		0.5
	};
	const quicknet::matrix_t weights{1,1,weights_a};

	float bias_a[1]{
		0.0
	};
	const quicknet::vector_t bias{1,bias_a};

	float output_a[1]{
		0.0
	};
	quicknet::vector_t output{1,output_a};

	quicknet::Layer layer{weights, bias, output, nullptr};

	const std::vector<float> inputs{1.0, 3.5, 0.25, -1.0};

	for(float in : inputs) {
		float input_a[1]{
			in
		};

		const quicknet::vector_t input{1,input_a};
		const quicknet::vector_t& output = layer.feedForward(input);

		ASSERT_EQ(in * 0.5, output(0));
	}
}

TEST(LayerTest, FeedForwardSimpleBias) {
	float weights_a[1]{
		0.5
	};
	const quicknet::matrix_t weights{1,1,weights_a};

	float bias_a[1]{
		-0.75
	};
	const quicknet::vector_t bias{1,bias_a};

	float output_a[1]{
		0.0
	};
	quicknet::vector_t output{1,output_a};

	quicknet::Layer layer{weights, bias, output, nullptr};

	const std::vector<float> inputs{1.0, 3.5, 0.25, -1.0};

	for(auto in : inputs) {
		float input_a[1]{
			in
		};

		const quicknet::vector_t input{1,input_a};
		const quicknet::vector_t& output = layer.feedForward(input);

		ASSERT_EQ(in * 0.5 - 0.75, output(0));
	}
}

TEST(LayerTest, FeedForwardComplexIn) {
	float weights_a[2]{
		0.5, 2.25
	};
	const quicknet::matrix_t weights{1,2,weights_a};

	float bias_a[1]{
		-0.75
	};
	const quicknet::vector_t bias{1,bias_a};

	float output_a[1]{
		0.0
	};
	quicknet::vector_t output{1,output_a};

	quicknet::Layer layer{weights, bias, output, nullptr};

	const std::vector<std::vector<float>> inputs{{1.0, 3.5}, {0.25, -1.0}};

	for(auto in : inputs) {
		float input_a[2]{
			in[0],
			in[1]
		};

		const quicknet::vector_t input{2,input_a};
		const quicknet::vector_t& output = layer.feedForward(input);

		ASSERT_EQ(in[0] * 0.5 + in[1] * 2.25 - 0.75, output(0));
	}
}

TEST(LayerTest, FeedForwardComplexOut) {
	float weights_a[2]{
		0.5,
		2.0
	};
	const quicknet::matrix_t weights{2,1,weights_a};

	float bias_a[2]{
		-1.0, 1.0
	};
	const quicknet::vector_t bias{2,bias_a};

	float output_a[2]{
		0.0, 0.0
	};
	quicknet::vector_t output{2,output_a};

	quicknet::Layer layer{weights, bias, output, nullptr};

	const std::vector<float> inputs{1.0, 3.5, 0.25, -1.0};

	for(auto in : inputs) {
		float input_a[1]{
			in
		};

		const quicknet::vector_t input{1,input_a};
		const quicknet::vector_t& output = layer.feedForward(input);

		ASSERT_EQ(in * 0.5 - 1.0, output(0));
		ASSERT_EQ(in * 2.0 + 1.0, output(1));
	}
}

