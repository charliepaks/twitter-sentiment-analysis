
# Twitter Sentiment Analysis Project

### Overview
This project aims to classify tweets into three sentiment categories: negative (-1), neutral (0), and positive (+1). The dataset contains two fields: the tweet text and its corresponding sentiment label. The goal is to build a model that can accurately predict the sentiment of a given tweet.

### Dataset
Sentiment Classes:
Negative: -1
Neutral: 0
Positive: +1
Fields:
Tweet: The text of the tweet.
Label: The sentiment associated with the tweet.

Source: Kaggle

## Running Locally

Add PYTHONPATH variable for `~/.bash_profile ` for MacOS


## Virtual Environment
Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version
```python
virtualenv --version
```

Create virtual environment

```python
virtualenv ml_package
```

Activate virtual environment

For Linux/Mac
```python
source ml_package/bin/activate
```
For Windows
```python
ml_package\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```


## Directory structure

```bash
prediction_model


├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


# Build the Package

1. Goto Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python prediction_model/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`

# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

2. Normal installation
```python
pip install .
```
```.``` refers to current directory

3. Also can be installed from git as well after pushing to github

```
pip install git+https://github.com/charliepaks/twitter-sentiment-analysis.git
```

# Testing the Package Working

1. Remove the PYTHONPATH from environment variables 
2. Goto a separate location which is outside of package directory
3. Create a new virual environment using the commands mentioned above & activate it
4. Before installing, test whether you are able to import the package of `prediction_model` - (you should not be able to do it)
5. Now in the new environment install the package using the generated file
`pip install git+https://github.com/charliepaks/twitter-sentiment-analysis.git`
6. Now try importing the prediction_model, you should be able to do it successfully
7. Extras : Run training pipeline using the package, and also conduct the test


