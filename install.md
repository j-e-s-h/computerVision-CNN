# objectDetection-CNN guide installation

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

## Create environment

```bash
conda env create -f environment.yml
activate objectdetection_cnn
```

## Set up project's module

To move beyond notebook prototyping, all reusable code should go into the project module . To use that package inside your project, install the project's module in editable mode, so you can edit files in the folder and use the modules inside your notebooks :

```bash
pip install --editable .
```

To use the module inside your notebooks, add `%autoreload` at the top of your notebook :

```python
%load_ext autoreload
%autoreload 2
```