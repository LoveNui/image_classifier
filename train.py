import argparse
import os

from model import create_model
from preprocess import augment_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', required=True, type=str,
        help=('path to dataset'))

    parser.add_argument(
        '--epochs', type=int,
        help=('number of epochs'))

    parser.add_argument(
        '--model_dir', required=True, type=str,
        help=('path where weights will be saved'))
    args = parser.parse_args()
    return args

def train():
    '''Training model'''

    args = parse_args()

    training_set, validation_set = augment_images(args.path, 'train')
    model = create_model(training_set.num_classes)

    model.compile(optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["acc"])

    model.fit(training_set,
              validation_data=validation_set,
              epochs=args.epochs)

    model.save(os.path.join(args.model_dir, 'weights.h5'))

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()