from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem import WordNetLemmatizer

#============================================================================#

wnl = WordNetLemmatizer()
tfidf = TfidfVectorizer()
encoder = LabelEncoder()

#============================================================================#

class SMSModel:
    def __init__(self, file_name, classifier):
        self.data = pd.read_csv(file_name)
        self.classifier = classifier
        self.corpus = []
        
    def explore_data(self):
        print(f'shape: \n{self.data.shape}', end='\n\n')
        print(f'head: \n{self.data.head()}', end='\n\n')
        print(f'describe: \n{self.data.describe().T}', end='\n\n')
        print(f'null count: \n{self.data.isnull().sum()}', end='\n\n')
        
    def prepare_data(self, range):
        self.data.drop([self.data.columns[col] for col in range], axis=1, inplace=True)
        self.data.columns = ['label', 'text']
        self.data.drop_duplicates(keep='first', inplace=True)  
        self.data['charnum'] = self.data['text'].apply(len)
        self.data['wordnum'] = self.data.apply(lambda ent: nltk.word_tokenize(ent['text']), axis=1).apply(len)
        
        print(self.data[:20])
        print(self.data.shape)
        
    def drop_redundant(self, limit):
        self.data = self.data[self.data['charnum'] < limit]
        print('Redundant data removed. shape: {}'.format(self.data.shape))
    
    def visualize_data(self):
        if (self.data['label'] is not None) and (self.data['text'] is not None):
            self.class_chart()
            self.scatter_chart()
            self.scatter_labels_chart()   
        else:
            print('You need to prepare your data first! Use prepare_data()')
            
    def prepare_text(self):
        if (self.data['label'] is not None) and (self.data['text'] is not None):
            self.data['text'] = self.data['text'].apply(self.clean_text)
            self.data['tokens'] = self.data.apply(lambda ent: nltk.word_tokenize(ent['text']), axis=1)
            self.data['lemmas'] = self.data['tokens'].apply(self.lemmatize_text)
            
            for ent in self.data['lemmas']:
                message = ' '.join([word for word in ent])
                self.corpus.append(message)
            print('Text prepared.\n{}'.format(self.corpus[:20]))
        else:
            print('You need to prepare your data first! Use prepare_data()')
            
    def split_data(self):
        if self.corpus:
            X = tfidf.fit_transform(self.corpus).toarray()
            y = encoder.fit_transform(self.data['label'])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, stratify=y, test_size=0.2, random_state=42
            )
            print('Data split successfully.')
        else:
            print('You need to prepare your text data first! Use prepare_text()')
            
    def train_model(self):
        if np.any(self.X_train):
            self.classifier.fit(self.X_train, self.y_train)
            
            print('Train set accuracy for {}: {:.3f}%'.format(type(self.classifier).__name__, 100*self.classifier.score(self.X_train, self.y_train)))
            print('Test set accuracy for {}: {:.3f}%'.format(type(self.classifier).__name__, 100*self.classifier.score(self.X_test, self.y_test)))
            print(classification_report(self.y_test, self.classifier.predict(self.X_test), target_names=['ham', 'spam']))
            self.conf_matrix_chart()
        else:
            print('You really wanted to train a model on completely unprepared data? Use split_data()')
    
    def class_chart(self):
        plt.figure(figsize=(15,10))
        count_label = pd.value_counts(self.data['label'], sort=True)
        count_label.plot(kind='bar', color=['blue', 'orange'])        
        plt.xlabel('label')
        plt.ylabel('count')
        plt.show()
    
    def scatter_chart(self):
        fig = pd.plotting.scatter_matrix(self.data, figsize=(15,15), marker='o', alpha=0.7)
        plt.show()
        
    def scatter_labels_chart(self):
        hamfig = pd.plotting.scatter_matrix(self.data[self.data.label == 'ham'], figsize=(15,15), marker='o', alpha=0.7, color='blue')
        spamfig = pd.plotting.scatter_matrix(self.data[self.data.label == 'spam'], figsize=(15,15), marker='o', alpha=0.7, color='orange')
        plt.show()
        
    def conf_matrix_chart(self):
        cmd = ConfusionMatrixDisplay.from_estimator(self.classifier, self.X_test, self.y_test, cmap=ListedColormap(['orange', 'blue']))
        plt.show()
                
    def clean_text(self, text):
        copy = re.sub(r'[^\w\s]', '', text.lower())
        copy = copy.lower()
        copy = copy.split()
        sw = stopwords.words('english')
        copy = ' '.join(x for x in copy if x not in sw and x != '_')
        return copy    
        
    def lemmatize_text(self, text):
        lemmas = [wnl.lemmatize(word, pos='v') for word in text]    
        return lemmas
    
    def __del__(self):
        print(f'{self} wiped from memory.')