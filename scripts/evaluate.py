#!/usr/bin/env python3
"""Evaluation script."""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to evaluation data")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")

if __name__ == "__main__":
    main()