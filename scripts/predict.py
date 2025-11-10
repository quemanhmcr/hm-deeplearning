#!/usr/bin/env python3
"""Inference script."""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input data/file")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    print(f"Running inference with: {args.checkpoint}")
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")

if __name__ == "__main__":
    main()