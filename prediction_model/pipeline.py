from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.linear_model import RandomForestClassifier
import numpy as np

classification_pipeline = Pipeline(
    [
        
        ('RandomForestClassifier',RandomForestClassifier(random_state=0))
    ]
)