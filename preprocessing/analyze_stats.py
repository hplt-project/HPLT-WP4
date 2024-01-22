import argparse
import os
import io
import gzip
import zstandard as zstd
import math
import json
from tqdm import tqdm
from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()


def analyze(input_file, output_plot_file, output_txt_file):
    with open(input_file, 'r') as f:
        clean_scores = json.load(f)

    clean_scores = np.array(clean_scores)

    # plot histogram, x axis ranges from 0 to 1
    plt.clf()
    plt.hist(clean_scores, bins=np.arange(0, 1, 0.01))
    plt.savefig(output_plot_file)

    # write stats
    with open(output_txt_file, 'w') as f:
        power_range = range(5, 155, 5)
        for power in tqdm(power_range):
            power = power / 10
            random.seed(42)
            sampled_scores = []
            n_accepted = 0
            for score in clean_scores:
                if random.random() < math.pow(score + 0.2, power):
                    n_accepted += 1
                    sampled_scores.append(score)
            
            f.write(f'power: {power}, n_accepted: {n_accepted}, ratio: {n_accepted / len(clean_scores)}\n')
            print(f'power: {power}, n_accepted: {n_accepted}, ratio: {n_accepted / len(clean_scores)}', flush=True)

    sampled_scores = np.array(sampled_scores)
    plt.clf()
    plt.hist(sampled_scores, bins=np.arange(0, 1, 0.01))
    plt.savefig(output_plot_file.replace('.png', '_sampled.png'))


if __name__ == "__main__":
    args = parse_args()

    output_plot_file = args.output_file + '.png'
    output_txt_file = args.output_file + '.txt'

    analyze(args.input_file, output_plot_file, output_txt_file)
