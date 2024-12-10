# computerVision-CNN

By: J. E. Salgado-Hernández

Notebooks where distinct arquitectures are used to solve problems of Object Detection (ResNet50) and Image Segmentation (U-Net).



## Environment Installation


### Requirements
```
* python 3.9
* opencv-python-headless 4.7.0.72
* tensorflow (installed with tensorflow-models)
```

The libraries needed for this project are in the `environment.yml` file. For an easy virtual environment creation, use the following command using **conda**:
```
conda env create -f environment.yml
```


### Model Repository and Needed Modifications
In order to install [**Object Detection API from Tensorflow 2**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), in the root of the repository run:
```
# Make sure you're in the root of the repository
cd models
# Create the directory for TensorFlow Modules
mkdir tf_models
cd tf_models
# Clone the repository
git clone https://github.com/tensorflow/models
```

After the model directory is in the repository, go to the file `models/tf_models/research/object_detection/packages/tf2/setup.py`. Some changes has to be done in order to make the `object_detection` module works properly. **Between lines 21 and 23** this has to be changed from
```
    'tf-models-official>=2.5.1',
    'tensorflow_io',
    'keras',
```
to
```
    'tf-models-official>=2.5.1, <2.16.0',
    'tensorflow_io',
    # 'keras',
```

Checkpoint used at the `resnet50_object_detection.ipynb` nootebook can be obtained via [Kaggle](https://www.kaggle.com/datasets/nobatgeldi/ssd-resnet50-v1-fpn-640x640-coco17-tpu8) as well. This happens because in recent versions of **Object Detection API from Tensorflow 2** the weights of the ResNet50 architecture were removed. Just download the checkpoints and place the directory in the following route: `models/tf_models/research/object_detection/test_data/checkpoint/`.


### Install `object_detection` module into the virtual environment
For an easy installation, run `tf_model_installation.sh` in your **linux terminal**:
```
. tf_model_installation.sh
```
It contains the neccesary commands to install the object detection module into the virtual environment, it also installs another libraries or framework such as tensorflow.

Or well, it can be done via **linux terminal** with tthe following lines:
```
# Activate venv if needed
conda activate computervision_cnn

cd models/tf_models/research/
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```



## Data

### Self-Driving Cars Dataset - *Object Detection Notebook*
Dataset was obtained via [Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars).

One way to dowload this dataset from the notebook is with the Kaggle API (first, the dependency has to be downloaded):
```
import kagglehub
path = kagglehub.dataset_download("alincijov/self-driving-cars")
```


### - CamSeq 2007 (Semantic Segmentation) *Image Segmentation Notebook*
Dataset was obtained via [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camseq-semantic-segmentation).

One way to dowload this dataset from the notebook is with the Kaggle API (first, the dependency has to be downloaded):
```
import kagglehub
path = kagglehub.dataset_download("carlolepelaars/camseq-semantic-segmentation")
```



## Notebooks
* `resnet50_object_detection.ipynb` fouces on the TensorFlow Models API. In particular uses the `object_detection` module and a pretrained model of **ResNet50**, trained with the [COCO Dataset](https://cocodataset.org/#home). **Note**. This notebook is complex to run due to lack of support for the tensorflow module ([README.md](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md) from the repository for more information). However, the section **Environment Installation** of this *README.md* file tells everything step by step how to make everything to work properly without having troubles with versions or data.
* `unet_semantic_segmentation` works with TensorFlow2 and Keras to build a U-Net arquitecture from scratch.



## Project Organization
```
├── LICENSE
├── .gitignore
├── environment.yml             <- The requirements file for reproducing the analysis environment.
├── README.md                   <- The top-level README for developers using this project.
├── install.md                  <- Detailed instructions to set up the virtual
|                                   environment of this project.
├── tf_model_installation.sh    <- File to install the Object Detection API from Tensorflow 2.
│
├── data                        
│   ├── raw                     <- The original, immutable data dump.
│   ├── interim                 <- Intermediate data that has been transformed.
│   └── processed               <- The final, canonical data sets for modeling.
│
├── models                      <- Models directory
│   ├── tf_models               <- 'object_detection' module from Tensorflow2 (empty at first).
│   └── u_net                   <- Weights checkpoints for the U-Net training.
│
└── notebooks                   <- Jupyter notebooks.
```


## License

MIT