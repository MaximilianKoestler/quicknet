#!/usr/bin/env python3

import json

from argparse import ArgumentParser

def main(args):
    with open(args.input) as f:
        network = json.load(f)

    indent = ' ' * 4
    code_arrays = []
    code_layers = []
    num_layers = len(network['layers'])

    code_layers.append('static quicknet::Layer layers[%d] {' % (num_layers))
    for i, layer in enumerate(network['layers']):
        code_arrays.append('/******** Layer %d ********/' % i)

        code_arrays.append('static const float l%d_weights_array[%d] {' % (i, layer['inputs'] * layer['outputs']))
        for row in layer['weights']:
            code_arrays.append(indent + ','.join(['%.8f' % v for v in row]) + ',')
        code_arrays.append('};')

        code_arrays.append('static float l%d_bias_array[%d] {' % (i, layer['outputs']))
        code_arrays.append(indent + ','.join(['%.8f' % v for v in layer['bias']]))
        code_arrays.append('};')

        code_arrays.append('static float l%d_output_array[%d];' % (i, layer['outputs']))
        code_arrays.append('')

        code_arrays.append('static const quicknet::matrix_t l%d_weights{%d,%d,l%d_weights_array};' % (i, layer['outputs'], layer['inputs'], i))
        code_arrays.append('static const quicknet::vector_t l%d_bias{%d,l%d_bias_array};' % (i, layer['outputs'], i))
        code_arrays.append('static quicknet::vector_t l%d_output{%d,l%d_output_array};' % (i, layer['outputs'], i))
        code_arrays.append('')

        code_layers.append(indent + '{l%d_weights, l%d_bias, l%d_output, quicknet::quick_%s},' % (i, i, i, layer['activation']))

    code_layers.append('};')
    code_layers.append('')

    code_cpp = ''
    code_cpp += '\n'.join(code_arrays) + '\n'
    code_cpp += '/******** Network ********/\n'
    code_cpp += '\n'.join(code_layers) + '\n'
    code_cpp += 'NeuralNetwork::NeuralNetwork() : network{%d, layers} {\n}\n' % num_layers

    with open(args.output, 'w') as f:
        f.write(code_cpp)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('output', type=str, help='Output file')

    main(parser.parse_args())

