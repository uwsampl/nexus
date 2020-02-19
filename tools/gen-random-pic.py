#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.dirname, exist_ok=True)
    for i in tqdm(range(args.number)):
        filename = os.path.join(args.dirname, f'random{i:04d}.jpg')
        cv2.imwrite(filename, np.floor(np.random.random((args.height, args.width, 3)) * 255).astype(np.uint8))


if __name__ == '__main__':
    main()

