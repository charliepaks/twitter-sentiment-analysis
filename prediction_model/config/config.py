import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

FILE_NAME = 'Twitter_Data.csv'


MODEL_NAME = 'model.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')



TARGET = 'category'

#Final features used in the model
FEATURES = ['clean_text', 'category']

PRED_FEATURES = ['clean_text']

STEMMED_DATA = ['stemmed_content']
CLEAN_TEXT = ['clean_text']





