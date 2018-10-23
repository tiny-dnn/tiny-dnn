#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import torch


nets = {
    1: ["vgg.0", "vgg.2", "vgg.5", "vgg.7", "vgg.10", "vgg.12", "vgg.14", "vgg.17", "vgg.19", "vgg.21"],
    2: ["vgg.24", "vgg.26", "vgg.28", "vgg.31", "vgg.33"],
    3: ["extras.0", "extras.1"],
    4: ["extras.2", "extras.3"],
    5: ["extras.4", "extras.5"],
    6: ["extras.6", "extras.7"],
    7: ["loc.0"],
    8: ["loc.1"],
    9: ["loc.2"],
    10: ["loc.3"],
    11: ["loc.4"],
    12: ["loc.5"],
    13: ["conf.0"],
    14: ["conf.1"],
    15: ["conf.2"],
    16: ["conf.3"],
    17: ["conf.4"],
    18: ["conf.5"],
}


def dump_layer_weights(f, weight, bias):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for k in range(weight.shape[2]):
                for m in range(weight.shape[3]):
                    f.write('%.24f ' % weight[i][j][k][m])
    _ = f.write('\n')
    for i in range(bias.shape[0]):
        f.write('%.24f ' % bias[i])
    _ = f.write('\n')



def dump_net_weights(model_path, output_folder):
    ckpt = torch.load(model_path)

    for net_id in nets:
        layers = nets[net_id]
        output_file_path = os.path.join(output_folder, '%02d.weights' % net_id)
        
        print('Saving weights to %s' % output_file_path)
        with open(output_file_path, 'w') as f:
            for layer in layers:
                weight = ckpt['%s.weight' % layer]
                bias = ckpt['%s.bias' % layer]
                dump_layer_weights(f, weight, bias)


def main():
    if not len(sys.argv) == 3:
        print('python convert_models.py model_path output_folder')
        sys.exit(1)

    model_path = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(model_path) or os.path.isdir(model_path):
        print('[ERROR] Model file not exists!')
        sys.exit(2)
    if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
        print('[ERROR] Output folder not exists!')
        sys.exit(2)

    dump_net_weights(model_path, output_folder)


if __name__ == '__main__':
    main()
