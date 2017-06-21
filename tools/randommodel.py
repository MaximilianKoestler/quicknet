#!/usr/bin/env python3

import json
import random

from argparse import ArgumentParser, RawDescriptionHelpFormatter

def weight():
    return random.random() * 2.0 - 1.0

def main(args):
    inputs, *layers = args.network.split('/')
    inputs = int(inputs)

    network = {}
    network['layers'] = []

    for l in layers:
        layer = {}

        outputs, activation = l.partition(',')[::2]
        outputs = int(outputs)
        if activation == '':
            activation = 'none'

        layer['activation'] = activation
        layer['inputs'] = inputs
        layer['outputs'] = outputs

        layer['bias'] = [weight() for j in range(0, outputs)]
        layer['weigts'] = [[weight() for i in range(0, inputs)] for j in range(0, outputs)]

        network['layers'].append(layer)

        inputs = outputs

    string = json.dumps(network, indent=4, sort_keys=True)
    if args.output != None:
        with open(args.output, 'w') as f:
            f.write(string)
    else:
        print(string)

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, epilog='''
example usage:
    ./randommodel.py 5/3,sigmoid/2,sigmoid

    ./randommodel.py 1/2 random.net
''')

    parser.add_argument('network', type=str, help='network string inputs(/outputs(,activation)?)+')
    parser.add_argument('output', type=str, nargs='?', help='output file [optional]')

    main(parser.parse_args())

