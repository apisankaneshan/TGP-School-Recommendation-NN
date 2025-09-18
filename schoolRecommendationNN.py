import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import pandas as pd

df = pd.read_csv("schoolReviews.csv")
print(df.head())
