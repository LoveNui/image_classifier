import argparse
import glob
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from preprocess import augment_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', required=True,
        help=('path, image from dataset'))

    parser.add_argument(
        '--model', required=True, type=str,
        help=('path to model'))

    args = parser.parse_args()
    return args


def inference():
    '''Inference on a single image

    Returns:
        name of predicted class
    '''
    args = parse_args()

    list_classes = []

    test_image = load_img(args.image, target_size = (224, 224))
    test_image = img_to_array(test_image)

    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    test_image = preprocess_input(test_image)

    model = load_model(args.model)

    result = model.predict(test_image)[0].argmax()

    rootdir = os.path.dirname( os.path.dirname(args.image))
    for path in glob.glob(f'{rootdir}/*'):
        list_classes.append(os.path.basename(path))

    class_names = sorted(list_classes)
    name_id_map = dict(zip(class_names, range(len(class_names))))

    labels = list(name_id_map.items())

    for label, i in labels:
        if i == result:
            return label

def main():
    result = inference()
    print(f'The image is {result}')

if __name__ == '__main__':
    main()