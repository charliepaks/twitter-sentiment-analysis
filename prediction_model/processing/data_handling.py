import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
pattern =re.compile('[^a-zA-Z]')
english_stopwords = stopwords.words('english')
port_stemmer = PorterStemmer()




PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config



#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] # Fix Column names
    return _data[config.FEATURES]



def preprocessed_text(text):
  # Initialize the lemmatizer and stopwords
  lemmatizer = WordNetLemmatizer()

  stemmed_content = re.sub(pattern,' ',text)
  stemmed_content = stemmed_content.lower()

  stemmed_content = stemmed_content.split()

  # Apply lemmatization instead of stemming and remove stopwords
  stemmed_content = [lemmatizer.lemmatize(word) for word in stemmed_content if word not in english_stopwords]
  stemmed_content = ' '.join(stemmed_content)


  return stemmed_content

   

    

# Separate X and y
def separate_data(data):
    data[config.STEMMED_DATA] = data[config.CLEAN_TEXT].apply(preprocessed_text)
    cv = CountVectorizer()
    X = data[config.STEMMED_DATA]
    y= data[config.TARGET]
    X= cv.fit_transform(X)
    return X,y

def scale_X(X):
   sc = StandardScaler(with_mean=False)
   X = sc.fit_transform(X)
   return X

#Split the dataset
def split_data(X, y, test_size=0.2, random_state=42):
  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    print(save_path)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

#Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH,pipeline_to_load)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded