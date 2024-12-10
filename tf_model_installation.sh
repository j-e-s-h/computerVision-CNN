#!bin/bash

# Activate venv if needed
conda activate computervision_cnn

cd models/tf_models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .