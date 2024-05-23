import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#This functuion used to clean the text by seperating the text into words and removing unwanted symbol,number and space and return cleaned text
def cleaned_texts(texts):
  stop_words=set(stopwords.words('english'))
  clean_texts=[]
  for text in texts:   
    text=text.lower()
    tokens=word_tokenize(text)
    clean_tokens=[]
    lemmatizer=WordNetLemmatizer()
    for word in tokens:
       if (word.isalnum()) and (not word.isdigit()) and ( word not in stop_words):
        word=word.strip('.')
        word=word.strip()
        word=lemmatizer.lemmatize(word)
        clean_tokens.append(word)
    clean_text=' '.join(clean_tokens)
    clean_texts.append(clean_text)
  return clean_texts


#This function used to read the data and return as pandas dataframe
@st.cache_data
def read_data():
   df=pd.read_csv(r'C:\Users\berli\Downloads\archive\FinalBalancedDataset.csv')
   df=df.sample(n=10000)
   return df

#This function used for adding the given text in the dataframe  and return adding cleaned_text column in dataframe
def add_data(df,te):
   tx=[np.nan,np.nan,te]
   df.loc[df.index.max() + 1]=tx 
   texts=(df['tweet'].values)
   df['cleaned_texts']=cleaned_texts(texts)
   return df

#This function used to coverting the text data column into numberical value 
def bow(df):
    X=df['cleaned_texts']
    model=CountVectorizer()
    model.fit(X)
    output=pd.DataFrame(model.transform(X).toarray())
    output.columns=model.get_feature_names_out()
    return output

#This function used to coverting the text data column into numberical value 
def tfidf(df):
    X=df['cleaned_texts']
    model=TfidfVectorizer()
    model.fit(X)
    output=pd.DataFrame(model.transform(X).toarray())
    output.columns=model.get_feature_names_out()
    return output

#This function is used to training and testing models and display the given text is toxic or not
@st.cache_data
def model(output,df,models):
            X=output.iloc[:-1]
            y=df['Toxicity'].iloc[:-1]
            a =output.iloc[-1].to_frame().T
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if models=='Logistic Regression':
               model = LogisticRegression()

            elif models=='K-NN':
              model = KNeighborsClassifier()   

            elif models=='svm':
               model = svm.SVC()

            elif models=='Decision Tree Classifier':
               model = DecisionTreeClassifier()
         
            elif models =='Random Forest Classifier':
               model = RandomForestClassifier()

            elif models =='GaussianNB':
               model = GaussianNB()
            model.fit(X_train, y_train)           
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            test_predict1=model.predict(a)
            b=test_predict1.item()
            if b==1.0:
               st.write(":orange[The given text is Toxic]")
            elif b==0.0:
               st.write(":blue[The given text is Non-Toxic]")
            st.write("---------------------------------------------------------------")
            st.write(f":violet[Train Accuracy]-{accuracy_score(y_train,train_predict)}")
            st.write(f":violet[Train Precision]-{precision_score(y_train,train_predict)}")
            st.write(f":violet[Train F1] -{f1_score(y_train,train_predict)}")
            st.write(f":violet[Train Recall] -{recall_score(y_train,train_predict)}")
            st.write(f":violet[Train Auc] -{roc_auc_score(y_train,train_predict)}")
            st.write(f":violet[Train Confusion Matrix] -{confusion_matrix(y_train,train_predict)}")

            st.write(f":violet[Test Accuracy] -{accuracy_score(y_test,test_predict)}")
            st.write(f":violet[Test Precision]-{precision_score(y_test,test_predict)}")
            st.write(f":violet[Test F1 ]-{f1_score(y_test,test_predict)}")
            st.write(f":violet[Test Recall] -{recall_score(y_test,test_predict)}")
            st.write(f":violet[Test Auc] -{roc_auc_score(y_test,test_predict)}")
            st.write(f":violet[Test Confusion Matrix] -{confusion_matrix(y_test,test_predict)}")
            
          


df=read_data()
te=st.text_input(label="Text")
df=add_data(df,te)
ck=st.selectbox(label="select a bow or Tfidf",options=['bow','TF-IDF'])
if ck=='bow':
  output=bow(df)
else:
   output=tfidf(df)
models=st.selectbox(label="Models",options=['Logistic Regression','K-NN','svm','Decision Tree Classifier','Random Forest Classifier','GaussianNB'])
model(output,df,models)
