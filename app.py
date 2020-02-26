"""
Building and Deploying NLP Sentiment Analysis Model	  	   		     			  		 			     			  	  		 	  	 		 			  		  			

Project based on 
A to Z (NLP) Machine Learning Model building and Deployment.
https://www.udemy.com/course/a-to-z-nlp-machine-learning-model-building-and-deployment/

Original Repo:
https://github.com/mdrijwan123/TWR-programming-Repo

Changes:
- Process form inputs before passing to trained NLP model

To Do:
- Explore models besides Log. Regression.
- how to deal with synonyms and words not within the BOW model.
"""

from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer # natural-language-tool-kit for NLP
import re # regex
import string
from sklearn.feature_extraction.text import CountVectorizer # can also use tf-idf
from sklearn.linear_model import LogisticRegression # we can use other models


def remove_pattern(input_txt,pattern):
    """ 
    Remove any occurrences of pattern string from input text.

    Parameters: 
    input_txt (str) 
    pattern (str) 

    Returns: 
    input_txt (str): With any occurrence of pattern removed. 
    """
    r = re.findall(pattern,input_txt)

    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

def count_punct(text):
    """Counts percentage of punctuations in text"""
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

def process_text(input_txt):
    """ 
    Process text for model

    Parameters: 
    input_txt (str): Input text for sentiment analysis.
    # to be implemented - input_txt as dataframe

    Returns: 
    tokenized_txt (str): Post-tokenize and post-stemming text.
    """
    # if input is string
    tidy_txt = remove_pattern(input_txt,"@[\w]*")
    ##=============================== if input is dataframe ====================##
    # tidy_txt = np.vectorize(remove_pattern)(input_txt,"@[\w]*")                #
    ##==========================================================================##
    # remove special characters
    tidy_txt = tidy_txt.replace("[^a-zA-Z#]"," ")
    # split into words
    tokenized_txt = tidy_txt.split()
    # perform stemming
    stemmer = PorterStemmer()
    tokenized_txt = [stemmer.stem(i) for i in tokenized_txt]
    print(tokenized_txt)
    # joining words back
    tokenized_txt = ' '.join(tokenized_txt)
    return tokenized_txt


app = Flask(__name__)
data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["label","body_text"]

# Features and Labels
data['label'] = data['label'].map({'pos': 0, 'neg': 1}) # maps label to numeric
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*") # uses func to remove punctuation
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split()) # tokenize tweets into words
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # apply stemming
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) # combine string
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_tweet']
y = data['label']
print(type(X))

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data and convert to numeric matrix
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Using Selected Classifier
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y) # fit trained model

# Flask setup
@app.route('/')
# home site
def home():
    return render_template('home.html')

# home site has a form that calls predict
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message'] # retrieve input data from API request form
        # to be implemented: uploading of csv file to process multiple tweets!
        
        # need to process text too!
        data = [process_text(message),]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000) # for docker to run out of localhost and be accessible