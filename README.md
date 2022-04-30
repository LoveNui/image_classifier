# Requirements  
* Python>=3.7.0  
* tensorflow==2.8.0  
* pip install -r requirements.txt

# Dataset
The dataset is available on download [here](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

# Training
`python train.py --path path/to/dataset --epochs 12 --model_dir path/for/model`
### Parameters:
* path: path to dataset  
* epochs: number of epochs  
* model_dir: path where weights will be saved

# Testing
`python test.py --path path/to/dataset --model path/for/trained/model`
### Parameters:
* path: path to dataset  
* model: path to model

# Inference
`python inference.py --image path/to/image --model path/for/trained/model`
### Parameters:
* image: path, image from dataset  
* model: path to model

# 2D images projections
`python image_projection.py --path path/to/data --model path/for/trained/model`
### Parameters:
* path: path to test data, or data which will be projected   
* model: path to model

# Flask
`cd flask`  
`python run.py --path path/for/trained/model`
### Parameters:
* path: a required path to weights   
