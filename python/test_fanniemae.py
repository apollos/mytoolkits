from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
a = datetime.now()
data = pd.read_csv("./aci2013_sample.csv", delimiter=",")
b = datetime.now()
c = b - a
print ("Read csv spend %d Second and %d microseconds" % (c.seconds, c.microseconds))

dataset = data.loc[:, ["CLTV", "minfico", "backend", "dq90_12"]].dropna()
train, test = train_test_split(dataset, test_size=0.2)
# split into input (X) and output (Y) variables
X_train = np.array(train.loc[:,["CLTV", "minfico", "backend"]])
Y_train = np.array(train.loc[:,["dq90_12"]])
X_test = np.array(test.loc[:,["CLTV", "minfico", "backend"]])
Y_test = np.array(test.loc[:,["dq90_12"]])

# create model
model = Sequential()
model.add(Dropout(0.2, input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
a = datetime.now()
model.fit(X_train, Y_train, epochs=3, batch_size=500)
b = datetime.now()
c = b - a
print ("Training spend %d Second and %d microseconds" % (c.seconds, c.microseconds))

# evaluate the model
a = datetime.now()
scores = model.evaluate(X_test, Y_test)
b = datetime.now()
c = b - a
print ("Evaluation spend %d Second and %d microseconds" % (c.seconds, c.microseconds))

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
