import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


width = 4000
height = 3000
max_dim = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', required=True,
        help=('path to test data, or data which will be projected'))

    parser.add_argument(
        '--model', required=True, type=str,
        help=('path to model'))

    args = parser.parse_args()
    return args


def tsne_projection():
    '''Create 2D image projection with t-SNE'''

    args = parse_args()

    test_dir = Path(args.path)
    test_filepaths = list(test_dir.glob(r'*/*.*'))

    labels = [str(test_filepaths[i]).split('\\')[-2] for i in range(len(test_filepaths))]

    filepath = pd.Series(test_filepaths, name = 'Filepath').astype(str)
    labels = pd.Series(labels, name = 'Label')

    df_test = pd.concat([filepath, labels], axis =1)

    df_test = df_test.sample(frac = 1).reset_index(drop = True)

    test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )

    test_images = test_datagen.flow_from_dataframe(
        dataframe = df_test,
        x_col = 'Filepath',
        y_col = 'Label',
        target_size = (224, 224),
        color_mode = 'rgb',
        class_mode = 'categorical',
        shuffle = False )

    model = load_model(args.model)

    feat_extractor = Model(inputs=model.input,
                        outputs=model.layers[-2].output)
    features = feat_extractor.predict(test_images)


    tsne = TSNE().fit_transform(features)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(df_test['Filepath'].tolist(), tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.show()


def main():
    tsne_projection()

if __name__ == '__main__':
    main()

