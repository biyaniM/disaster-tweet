#%%
from operator import sub
from numpy.lib.function_base import vectorize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
# %%
#* Data input
df_train=pd.read_csv("../../data/train.csv")
df_test=pd.read_csv("../../data/test.csv")
# %%
#* Exploring missing values
print('In Training...')
for cols in df_train.columns:
    x=len(df_train[df_train[cols].isnull()==True])/len(df_train[cols])
    if x>0: print('\t',cols,round(x*100,2),'% missing values')
print('In Test...')
for cols in df_test.columns:
    x=len(df_test[df_test[cols].isnull()==True])/len(df_test[cols])
    if x>0: print('\t',cols,round(x*100,2),'% missing values')
#* Location can be dropped (intutively as well)
plt.barh(y=df_train.location.value_counts()[:15].index,width=df_train.location.value_counts()[:15])
for ind,val in enumerate(df_train.location.value_counts()[:15]): plt.text(val,ind,str(val))
plt.xticks()
plt.show()
df_train=df_train.drop(['location'],axis=1)
# %%
#* Compiling characteristics of text useful for labels
# Text length
df_train['text_length']=df_train.text.apply(lambda x: len(x))
df_train['word_count']=df_train.text.apply(lambda x: len(nltk.word_tokenize(x)))
fig,(ax1,ax2)=plt.subplots(1, 2, sharex=True,figsize=(12,6))
ax1.hist(x=df_train[df_train.target==1].text_length,label='Disaster Tweets',color='red')
ax2.hist(x=df_train[df_train.target==0].text_length,label='Non Disaster Tweets',color='yellow')
ax1.grid(); ax2.grid(); fig.legend()
fig.suptitle('Text Length Distribution')
plt.show()
fig,(ax1,ax2)=plt.subplots(1, 2, sharex=True,figsize=(12,6))
ax1.hist(x=df_train[df_train.target==1].word_count,label='Disaster Tweets',color='red')
ax2.hist(x=df_train[df_train.target==0].word_count,label='Non Disaster Tweets',color='yellow')
ax1.grid(); ax2.grid(); fig.legend()
fig.suptitle('Word Count Distribution')
plt.show()
#%%
#* Cleaning - Tokenizing, removing stopwords, punctuation,etc
stopwords = nltk.corpus.stopwords.words('english')
def remove_internet_jargon(text:str)->str:
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # removing URLs
    text = re.sub('<.*?>+', '', text) # removing HTML jargon
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'',text)
    return ''.join([i for i in text if not i.isdigit()])

df_train['text_cleaned']=df_train.text.apply( lambda x:
    remove_internet_jargon(' '.join(
        [nltk.stem.WordNetLemmatizer().lemmatize(word.lower()) for word in nltk.RegexpTokenizer(r'\w+').tokenize(x) if word not in stopwords]
    ))
)
df_test['text_cleaned']=df_test.text.apply( lambda x:
    remove_internet_jargon(' '.join(
        [nltk.stem.WordNetLemmatizer().lemmatize(word.lower()) for word in nltk.RegexpTokenizer(r'\w+').tokenize(x) if word not in stopwords]
    ))
)
df_train.head()
#%%
#* Model - vectorization and fitting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

vector = TfidfVectorizer()
train_cv = vector.fit_transform(df_train.text_cleaned)
test_cv = vector.transform(df_test.text_cleaned)

X_train, X_test, y_train, y_test = train_test_split(train_cv,df_train.target,test_size=0.2,random_state=2020)
print('X_train shape',X_train.shape)
print('X_test shape',X_test.shape)

#* Fitting into RF Classifier
RDClassifier = RandomForestClassifier(n_estimators=100,random_state=0)
RDClassifier.fit(X_train,y_train)

#* Prediction
y_pred = RDClassifier.predict(X_test)
# %%
#* Evaluation of model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
#%%
#* Saving RF model as pickle object
import pickle
with open('rf_text_classifier', 'wb') as picklefile:
    pickle.dump(RDClassifier,picklefile)
#%%
#* Predicting with same model
with open('rf_text_classifier', 'rb') as training_model: model = pickle.load(training_model)
y_pred2 = model.predict(test_cv)
# %%
#* Submission
sample_submission=pd.read_csv('../../data/sample_submission.csv')
submission_data = {"id":[],"target":[]}
for idx,pred in zip(sample_submission['id'].unique(),y_pred2): 
    submission_data['id'].append(idx)
    submission_data['target'].append(pred)
df_submission=pd.DataFrame.from_dict(submission_data)
df_submission
# %%
#* Converting to CSV
df_submission.to_csv('../submission.csv',index=False)