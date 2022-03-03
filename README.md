# Movie Reviews Sentiment Analysis

## Introduction
Movie reviews are an important way to measure a movie's performance. Providing a number/stars of a movie that can quantitatively suggest whether a movie is thriving or not. A series of movie reviews give us deeper qualitative insights into various aspects of the movie, meanwhile,  written reviews tell us about the film's strengths and weaknesses. An in-depth analysis of film reviews can inform whether the film generally satisfies audience expectations.

In this project, I will use three different types of deep neural networks: Densely Connected Neural Networks (Basic Neural Network), Convolutional Neural Networks (CNN) and Long Short-Term Memory Networks (LSTM), which is a variation of Recurrent Neural Networks.

## Library Used
```
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
```

## Technologies
This project is created with:
* Jupyter Notebook 6.0.3
