import argparse

from tensorflow.keras.models import load_model

from preprocess import augment_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        help=('path to dataset'))

    parser.add_argument(
        '--model',
        help=('path to model'))

    args = parser.parse_args()
    return args

def test():
    '''Evaluate model'''

    args = parse_args()

    test_generator= augment_images(args.path, 'test')

    model = load_model(args.model)
    results = model.evaluate(test_generator)

    print("test loss, test acc:", results)


def main():
    test()

if __name__ == '__main__':
    main()
