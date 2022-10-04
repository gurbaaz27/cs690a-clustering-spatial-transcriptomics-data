# Task 1

## Installation

Create a virtual environment, using `conda` or `venv`. Activate it and install the dependencies

```
conda create -n SpaGCN python=3.8
conda activate SpaGCN
cd SpaGCN/SpaGCN_package
python setup.py build
python setup.py install
cd ..
cd ..
pip install -r requirements.txt
```

## Usage

Run the python script (preferably on GPU) using

```
python cs690_task1_script.py
```

Alternatively, you can upload the script file on Colab, install the requirements and run the script with

```
%run cs690_task1_script.py
```
