import numpy as np
import csv
import argparse
from typing import Tuple

N = 1000000
L = 40
np.random.seed(46513465)


def generate(maxlen: int) -> Tuple[str, int]:
    if maxlen == 1:
        return '1', 1
    if np.random.randint(low=0, high=maxlen - 1):
        op = np.random.choice(['~', '|', '&'])
        if maxlen == 2:
            op = '~'
        if maxlen == 3:
            return '1' + '1' + op, 1
        if op == '~':
            gen = generate(maxlen - 1)
            return gen[0] + '~', 1 - gen[1]
        else:
            left_len = np.random.randint(1, maxlen - 2)
            right_len = maxlen - 1 - left_len
            left = generate(left_len)
            right = generate(right_len)
            if op == '|':
                ans = left[1] | right[1]
            else:
                ans = left[1] & right[1]
            return left[0] + right[0] + op, ans
    else:
        return '1', 1


def main(output_path: str, count: int, maxlen: int) -> None:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['expr', 'label'])
        for i in range(1, count):
            expr, label = generate(maxlen)
            writer.writerow([expr, label])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate boolean-expression dataset and save as CSV')
    parser.add_argument('--output', '-o', default='data.csv', help='Output CSV file path')
    parser.add_argument('--n', type=int, default=N, help='Number of examples to generate (default 1,000,000)')
    parser.add_argument('--maxlen', type=int, default=L, help='Max length parameter passed to generator (default 40)')
    args = parser.parse_args()
    main(args.output, args.n, args.maxlen)