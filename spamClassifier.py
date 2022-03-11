#%% Imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

#%% Data

# read in input file and create list from each line
with open('data/SMSSpamCollection','r') as f: 
    lines = f.readlines()

#%% Cleaning 

# split each line to create message type (ham 0 or spam 1) and messages
messageType = np.array([1 if message[:4].strip() == 'spam' else 0 for message in lines])
messages = [message[4:].strip() for message in lines]

# characters to remove from messages
specialChars = '",.<>:;/()*&^%$?!'

# removing the special characters above and extra spaces from strings
for i in range(len(messages)):
    for char in specialChars:
        messages[i] = messages[i].replace(char, ' ')
    messages[i] = ' '.join(messages[i].split())
    
#%% Feature Extraction

# vectorizing the words that have appeared in all messages 
vectorizer = TfidfVectorizer()
vectorizedMessages = vectorizer.fit_transform(messages)

# transform into an array and adding back the spam/ham markers
data = vectorizedMessages.toarray()
data = np.concatenate((messageType.reshape(len(messageType),1), data), 1)
 
#%% Splitting Data

# separating data into ham and spam sets
spam = np.array([data[i] for i in range(len(data)) if data[i,0] == 1.0])
ham = np.array([data[i] for i in range(len(data)) if data[i,0] == 0.0])

# look at the percentage of spam messages in the dataset
spamPercentage = len(spam) / (len(spam) + len(ham))

# split both spam and ham arrays into train test splits
spamXTrain, spamXTest, spamYTrain, spamYTest = train_test_split(spam[:,1:], spam[:,0], test_size = 0.25, random_state = 42)
hamXTrain, hamXTest, hamYTrain, hamYTest = train_test_split(ham[:,1:], ham[:,0], test_size = 0.25, random_state = 42)

# check that the spam percentage remains close to the original dataset
trainPercentage = len(spamXTrain) / (len(spamXTrain) + len(hamXTrain))

# combine spam and ham splits into overall splits
xTrain = np.concatenate((spamXTrain, hamXTrain), 0)
xTest = np.concatenate((spamXTest, hamXTest), 0)
yTrain = np.concatenate((spamYTrain, hamYTrain), 0)
yTest = np.concatenate((spamYTest, hamYTest), 0)

#%% Train Model

# create logistic regression model, train it, and create predictions from test set
model = LogisticRegression().fit(xTrain, yTrain)
predictions = model.predict(xTest)

#%% Metrics

# run different metrics to compare real values with the predictions from the model
accuracy = metrics.accuracy_score(yTest, predictions)
precision = metrics.precision_score(yTest, predictions)
recall = metrics.recall_score(yTest, predictions)
auc = metrics.roc_auc_score(yTest, predictions)
matrix = metrics.confusion_matrix(yTest, predictions)
