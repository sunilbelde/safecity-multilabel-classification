import numpy as np
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')

class Preprocessing():
  def __init__(self):
    self.glove_word_embeddings = pickle.load(open('small_glove_embeddings.pkl','rb'))
    self.glove_words=self.glove_word_embeddings.keys()
    
    self.stopWords=stopwords.words('english')
    self.stemmer=SnowballStemmer('english')

    #removing no,nor and not words from the english stopwords
    self.stopWords.remove('not')
    self.stopWords.remove('no')
    self.stopWords.remove('nor')
    
    self.sid = SentimentIntensityAnalyzer()

  def decontracted(self,text):
    '''Funtion to expand the sentences which are in short forms'''
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text
    
  def remove_special_chars(self,text):
    '''This function removes the special chars from the text'''
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text=text.lower()
    return text
  def remove_stopwords(self,text):
    '''This function removes the stopwords from the text'''
    text=[word for word in text.split() if not word in self.stopWords]
    text=' '.join(text)
    return text
    
  def stemming(self,text):
    '''This function is to do stemming on words of text'''
    text=' '.join([self.stemmer.stem(word) for word in text.split()])
    return text
  def preprocess_text(self,text):
    '''This function does all the text preprocessing steps and return a clean text'''
    text=self.decontracted(text)
    text=self.remove_special_chars(text)
    text=self.remove_stopwords(text)
    text=self.stemming(text)
    return text 
    
  def get_embedding_features(self,data,word_embeddings,model_words):
    '''This function takes dataframe as input and returns fasttext vecotr respresent of text data(Description)'''
    vector_rep=[]
    preprocessed_descriptions = data['Description'].values
    for text in preprocessed_descriptions: # For each description
      vector=np.zeros(300)
      n=0
      for word in text.split():# For each word in vector
        if (word in model_words):
          vec=word_embeddings[word] #Getting the word's w2v representation
          vector+=vec
          n+=1
      if n!=0:
        vector/=n
      vector_rep.append(vector)
    return np.array(vector_rep)

  def get_word_char_lengths(self,data):
    '''This function takes input dataframe and return with length of text by wordlevel and characterleve'''
    length_features=[]
    for index,row in data.iterrows():
      text=row['Description']
      length_wordlevel=len(text.split()) # Getting the number of words
      len_charlevel=len(text) # Getting the number characters including spaces
      length_features.append([length_wordlevel,len_charlevel])
    return pd.DataFrame(length_features,columns=['length_word_level','length_char_level'])
    
  def sentiment_score(self,data):
    '''This function takes dataframe as input and returns sentiment scores of text data'''
    sentiments=[]
    preprocessed_descriptions = data['Description'].values
    for text in preprocessed_descriptions:
      polarities=self.sid.polarity_scores(text) # Getting the sentiment scores of text
      sentiments.append(list(polarities.values()))
    return pd.DataFrame(sentiments,columns=['negative','neutral','positive','compound'])
    
  def get_vector_representation(self,query):
    #pre_query=self.preprocess_text(query)
    qd=pd.DataFrame([query],columns=['Description'])
    quer_fasttext=self.get_embedding_features(qd,self.glove_word_embeddings,self.glove_words)
    quer_fasttext=pd.DataFrame(quer_fasttext,columns=['embed_'+str(i) for i in range(300)])
    query_lengths=self.get_word_char_lengths(qd)
    qeury_sentiments=self.sentiment_score(qd)
 
    vector=pd.concat([quer_fasttext,query_lengths,qeury_sentiments],axis=1)

    vector=vector.values[0]
    return vector