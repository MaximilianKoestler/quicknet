#include <gtest/gtest.h>

#include "Layer.h"

TEST(MatrixTest, Initialization) {
	float a[2 * 3]{
		1,2,3,
		4,5,6
	};

	quicknet::matrix_t m{2,3,a};

	ASSERT_EQ(2, m.rows());
	ASSERT_EQ(3, m.columns());

	for(int j = 0; j < m.columns(); j++) {
		for(int i = 0; i <m.rows(); i++) {
			ASSERT_EQ(a[i * m.columns() + j], m(i,j));
		}
	}
}

TEST(VectorTest, Initialization) {
	float a[3]{
		1,2,3,
	};

	quicknet::vector_t m{3,a};

	ASSERT_EQ(3, m.length());

	for(int i = 0; i <m.length(); i++) {
		ASSERT_EQ(a[i], m(i));
	}
}
