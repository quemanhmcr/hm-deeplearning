#!/usr/bin/env python3
"""Training script."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"Training with config: {args.config}")
    if args.resume:
        print(f"Resuming from: {args.resume}")

if __name__ == "__main__":
    main()