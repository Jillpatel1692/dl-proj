
# Project Title

Reuters 21578

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Text Classification](#text-classification)
- [Topic Modeling](#topic-modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In summary, the data processing pipeline includes loading and preprocessing text data, splitting it for training and testing, vectorizing using TF-IDF, training a Naive Bayes classifier, optional shuffling and set splitting, optional tokenization for LDA, and optional LDA topic modeling with coherence score calculation. The pipeline aims to prepare and analyze text data for tasks like classification and topic modeling.

## Getting Started

### Prerequisites

# Install the required libraries
pip install numpy pandas scikit-learn nltk gensim

# Libraries for Text Classification
import numpy as np <br>
import pandas as pd <br>
from sklearn.feature_extraction.text import TfidfVectorizer <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.naive_bayes import MultinomialNB <br>
from sklearn.metrics import classification_report <br>
from nltk.corpus import reuters <br>
from nltk.corpus import stopwords <br>
from nltk.tokenize import word_tokenize <br>
from nltk.stem import PorterStemmer <br>

# Additional NLTK Downloads
import nltk 
nltk.download("reuters") <br>
nltk.download("stopwords") <br>
nltk.download("punkt") <br>

# Libraries for Data Set Splitting (Optional)
from sklearn.utils import shuffle

# Libraries for Topic Modeling (Optional)
import gensim
from gensim import corpora <br>
from gensim.models import CoherenceModel <br>

# Libraries for Document Categorization by Topics (Optional)
from collections import defaultdict


Results
The code accomplishes text classification using Naive Bayes, includes optional data set splitting, and offers optional topic modeling using LDA. Results include classifier evaluation, data set organization, topic discovery, and document categorization by topics. Users can adapt the code for their text analysis tasks.
