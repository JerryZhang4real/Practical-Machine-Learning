# main.py
import argparse
from train import train
from evaluate import evaluate

def main(args):
    if args.mode == 'train':
        print("Starting training...")
        train()
    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        evaluate()
    else:
        print("Invalid mode selected. Please choose 'train' or 'evaluate'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Main entry point for the deep learning pipeline. Choose a mode to run."
    )
    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'evaluate'],
        help="Mode to run: 'train' to train the model, 'evaluate' to run evaluation."
    )
    args = parser.parse_args()
    main(args)
