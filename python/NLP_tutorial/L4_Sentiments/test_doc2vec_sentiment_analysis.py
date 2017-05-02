"""
Created on Mon Apr 19 21:35:18 2017

@author: raysun

Source - https://radimrehurek.com/gensim/models/doc2vec.html
         https://github.com/linanqiu/word2vec-sentiments
         http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression ...
"""
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Step 1. Define a class to load data

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# Step 2. Load data. The result is to have five documents:
# test-neg.txt: 12500 negative movie reviews from the test data
# test-pos.txt: 12500 positive movie reviews from the test data
# train-neg.txt: 12500 negative movie reviews from the training data
# train-pos.txt: 12500 positive movie reviews from the training data
# train-unsup.txt: 50000 Unlabelled movie reviews

# Load input data
sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
sentences = TaggedLineSentence(sources)

# Step 3. Build Doc2Vec model

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

for epoch in range(10):
    model.train(sentences.sentences_perm(),epochs=model.iter,total_examples=model.corpus_count)

# Save the d2v model 
model.save('/tmp/d2v_imdb.model')

# Load the saved d2v model
model = Doc2Vec.load('/tmp/d2v_imdb.model')

# Step 4. Perform similarity analysis

# Check what words are similar to 'good'
print model.most_similar('good')

# Check what words are similar to 'bad'
print model.most_similar('bad')

# Inspect each of the vectors of the words and sentences in the model using model.syn0 
# Here, we don't want to use the entire syn0 since that contains the vectors for the words 
# as well, but we are only interested in the ones for sentences.
# Below is a sample vector for the first sentence in the training set for negative reviews:
print model.docvecs['TRAIN_NEG_0']

# Step 5. Create training data sets

train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

print train_labels

# Step 6. Create test data sets

# initiate a matrix for test data and a vector for test label (observation or output)
test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0
    
print test_labels

# Step 7. Classify sentiments using classifiers from the scikit-learn package

# Case 1: Logistic regression with L2 norm
# Initialate classifier using the logistic regressor with the default parameters
classifier_1 = LogisticRegression()

# Perform data fitting with the trained data
classifier_1.fit(train_arrays, train_labels)

# Run classifier using the logistic regressor with parameters to improve accuracy (L2 norm)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

# Compute prediction accuracy with the test data
print classifier_1.score(test_arrays, test_labels)   

# Case 2: Logistic regression with L1 norm
# Initialate classifier using the logistic regressor with the default parameters
classifier_2 = LogisticRegression()

# Perform data fitting with the trained data
classifier_2.fit(train_arrays, train_labels)

# Initialate classifier using the logistic regressor with parameters to improve accuracy (L1 norm)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001) 

# Compute prediction accuracy with the test data
print classifier_2.score(test_arrays, test_labels)

# Case 3 SGD classifier with L2 norm
# Initialate classifier using the logistic regressor with the default parameters
classifier_3 = SGDClassifier()

# Perform data fitting with the trained data
classifier_3.fit(train_arrays, train_labels)

# Run classifier using the SGD classifier with parameters to improve accuracy (L2 norm)
SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, 
              shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', 
              eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)

# Compute prediction accuracy with the test data
print classifier_3.score(test_arrays, test_labels)

# Case 4 Linear SVC classifier from SVM with L2 norm
# Initialate classifier using the logistic regressor with default parameters
classifier_4 = LinearSVC()

# Perform data fitting with the trained data
classifier_4.fit(train_arrays, train_labels)

# Run classifier using the liear SVC classifier with parameters to improve accuracy (L2 norm)
LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
          fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, 
          max_iter=1000)

# Compute prediction accuracy with the test data
print classifier_4.score(test_arrays, test_labels)

