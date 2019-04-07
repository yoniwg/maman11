## general imports
import random
import itertools
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

## project supplied imports
import exercise_data_builder
from abstract_solution_3 import AbstractSolution3

class Submission(AbstractSolution3):

    '''
    This is a class implementation exemplifying how to implement the class `AbstractSolution3` 
    for submission. We placed here the parts of from the exercise published notebook, 
    that are needed for training, as an example. You should do the same with your own code ―  
    organize it into the same interface spec given by the class `AbstractSolution3`.
    
    The submission scoring code will run your code through the method interfaces of `AbstractSolution3`.
    '''

    def __init__(self):
        pass # no-op


    def _add_to_ngrams(self, ngrams, max_ngram_len, token):

        ''' a helper function we use in our example implemenation
           (updates a pre-initialized ngrams dict with all ngrams present in the given input token). '''

        # per ngram length
        for n in range(1, max_ngram_len + 1):
            # sliding window iterate the token to extract its ngrams
            for idx in range(len(token) - n + 1):
                ngram = token[idx : idx + n]

                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

        return ngrams # return value used for test only


    def vectorize(self, word):
        ''' our sample function for turning a given token into a feature vector '''

        # ngram occurences
        vec1 = [0] * len(self.ngrams)
        for idx, ngram in enumerate(self.ngrams):
            if ngram in word:
                if vec1[idx]:
                    vec1[idx] += 1
                else:
                    vec1[idx] = 1

        # ngram occurences as prefix
        vec2 = [0] * len(self.ngrams)
        for idx, ngram in enumerate(self.ngrams):
            if word.startswith(ngram):
                if vec2[idx]:
                    vec2[idx] += 1
                else:
                    vec2[idx] = 1

        # ngram occurences as prefix
        vec4 = [0] * len(self.ngrams)
        for idx, ngram in enumerate(self.ngrams):
            if word[1:].startswith(ngram):
                if vec4[idx]:
                    vec4[idx] += 1
                else:
                    vec4[idx] = 1

        # ngram occurences as suffix
        vec3 = [0] * len(self.ngrams)
        for idx, ngram in enumerate(self.ngrams):
            if word.endswith(ngram):
                if vec3[idx]:
                    vec3[idx] += 1
                else:
                    vec3[idx] = 1

        vec5 = [0] * len(self.ngrams)
        for idx, ngram in enumerate(self.ngrams):
            if word[:-1].endswith(ngram):
                if vec5[idx]:
                    vec5[idx] += 1
                else:
                    vec5[idx] = 1

        # our feature vector here is a concatenation of 3 feature types:
        return vec1 + vec2 + vec3 + vec4 + vec5 + [len(word)]


    def train(self, tokens, y):

        max_ngram_len = 2

        # extract all ngrams from all corpus tokens
        self.ngrams = dict()
        for token in tokens:
            self._add_to_ngrams(self.ngrams, max_ngram_len, token)

        ## vectorizing the data
        X = list(map(self.vectorize, tokens))

        assert len(X) == len(y)

        # keep the following line uncommented to use Naive Bayes
        # model = MultinomialNB()

        # keep the following line uncommented to use Logistic Regression
        model = LogisticRegression(solver='liblinear', multi_class='auto')

        sample_weights = []

        for label in y:
            if label == 0:
                weight = 1
            elif label == 1:
                weight = 0.8
            elif label == 2:
                weight = 0.7
            sample_weights.append(weight)

        model.fit(X, y, sample_weights);

        return model
