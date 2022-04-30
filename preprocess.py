import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(dataset_path, mode):
    '''Add augmentation to dataset

    Params:
        dataset_path (str) : path to parent folder of train,test,validation
        mode (str) : augmentation for train or test

    Returns:
        training_set, validation_set / test_generator : data generator
    '''

    if mode=='train':

        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range= 20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')


        training_set = train_datagen.flow_from_directory(
            os.path.join(dataset_path, "train"),
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical")

        validation_set = train_datagen.flow_from_directory(
            os.path.join(dataset_path, "validation"),
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical")

        return training_set, validation_set

    elif mode=='test' :

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
            )

        test_generator = test_datagen.flow_from_directory(
            os.path.join(dataset_path, "test"),
            target_size=(224, 224),
            batch_size=1,
            class_mode="categorical")

        return test_generator