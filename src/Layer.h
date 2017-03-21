/*
 * quicknet
 *
 * An efficient implementation of artificial neural networks
 *
 * Authors: Maximilian Koestler <maximilian.koestler@tuhh.de>
 *
 * Copyright (c) 2017, Institute of Telematics, Hamburg University of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef QUICKNET__LAYER_H_
#define QUICKNET__LAYER_H_

#include "./Matrix.h"
#include "./Vector.h"

namespace quicknet {

typedef float weight_t;

typedef Matrix<weight_t> matrix_t;
typedef Vector<weight_t> vector_t;

typedef void (*activation_t)(vector_t&);

class Layer {
public:
    Layer(const matrix_t& weights, const vector_t& bias, vector_t& output, activation_t activation);

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    ~Layer() = default;

    vector_t& feedForward(vector_t& input);

private:
    const matrix_t& weights;
    const vector_t& bias;
    vector_t& output;
    const activation_t activation;
};

} /* namespace quicknet */

#endif /* QUICKNET__LAYER_H_ */
