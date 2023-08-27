import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        
        # get dictionary of all words, each mapping to [0,0]
        all_words = dict()
        for sms in X.values:
            for word in sms.split():
                all_words[word] = np.array([0,0])

        # loop through all sentences in our data
        for i, sms in enumerate(X.values):

            # look at each word in each sentence
            for word in sms.split():

                #count occurances of that word
                count = 0
                if word in sms.split():
                    count += 1
                
                # make sure to keep track if the sentence is spam or ham
                if y.values[i] == "spam":
                    col = np.array([count,0])
                if y.values[i] == "ham":
                    col = np.array([0, count]) 

                # add to dictionary
                all_words[word] += col
                
        # Create dataframe from dictionary
        data = pd.DataFrame.from_dict(all_words)
        data[" "] = ["spam","ham"]
        data = data.set_index(" ")

        self.data = data
                
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        
        
        num_spam_words, num_ham_words = self.data.sum(axis = 1).values

        P_spam = num_spam_words / (num_spam_words + num_ham_words) # 7.3
        P_ham = num_ham_words / (num_spam_words + num_ham_words) # 7.4

        spam_list = []
        ham_list = []

        
        for sms in X.values:
            split = sms.split()

            done = set()
            ham_prod = 1
            spam_prod = 1

            for word in split:
                # Don't want to double count words
                if word not in done:
                    done.add(word)
                    n = split.count(word)

                    if word in self.data.loc["ham"]:
                        P_ham_word = self.data.loc["ham"][word] / num_ham_words
                    else:
                        P_ham_word = 1 
                    
                    if word in self.data.loc["spam"]:
                        P_spam_word = self.data.loc["spam"][word] / num_spam_words
                    else:
                        P_spam_word = 1 


                    ham_prod *= P_ham_word**n
                    spam_prod *= P_spam_word**n
                
            ham_list.append(ham_prod * P_ham)
            spam_list.append(spam_prod * P_spam)
        return np.vstack((ham_list, spam_list)).T

            # self.data["in"]["spam"] = self.data.loc["spam"]["in"]
                 
    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        
        # Call predict_proba function
        predictions = self.predict_proba(X)

        # Get argmaxes and return array of ham for index 0 and spam for index 1
        maxs = np.argmax(predictions, axis = 1)
        return np.array(["ham" if g == 0 else "spam" for g in maxs])

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''

        num_spam_words, num_ham_words = self.data.sum(axis = 1).values

        P_spam = num_spam_words / (num_spam_words + num_ham_words) # 7.3
        P_ham = num_ham_words / (num_spam_words + num_ham_words) # 7.4
        
        spam_list = []
        ham_list = []

        
        for sms in X.values:
            split = sms.split()

            done = set()
            ham_sum = 0
            spam_sum = 0

            for word in split:
                # Don't want to double count words
                if word not in done:
                    done.add(word)
                    n = split.count(word)

                    if word in self.data.loc["ham"]:
                        P_ham_word = (self.data.loc["ham"][word] + 1) / (num_ham_words + 2)
                    else:
                        P_ham_word = 1
                    
                    if word in self.data.loc["spam"]:
                        P_spam_word = (self.data.loc["spam"][word] + 1) / (num_spam_words + 2)
                    else:
                        P_spam_word = 1 


                    ham_sum += n*np.log(P_ham_word)
                    spam_sum += n*np.log(P_spam_word)
            
            ham_list.append(ham_sum + np.log(P_ham))
            spam_list.append(spam_sum + np.log(P_spam))
        return np.vstack((ham_list, spam_list)).T

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''

        # Call predict_log_proba function
        predictions = self.predict_log_proba(X)

        # Get argmaxes and return array of ham for index 0 and spam for index 1
        maxs = np.argmax(predictions, axis = 1)
        return np.array(["ham" if g == 0 else "spam" for g in maxs])


class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels

        Returns:
            self: this is an optional method to train
        '''

        # n_i = number of times ith word occurs in class K
        # N_k = total number of words in class K
        spam_rates = dict() # mapping word to rate
        ham_rates = dict()

        # get dictionary of all words, each mapping to [0,0]
        all_words = dict()
        for sms in X.values:
            for word in sms.split():
                all_words[word] = np.array([0,0])
        

        # loop through all sentences in our data
        for i, sms in enumerate(X.values):

            # look at each word in each sentence
            for word in sms.split():

                #count occurances of that word
                count = 0
                if word in sms.split():
                    count += 1
                
                # make sure to keep track if the sentence is spam or ham
                if y.values[i] == "spam":
                    col = np.array([count,0])
                if y.values[i] == "ham":
                    col = np.array([0, count]) 

                # add to dictionary
                all_words[word] += col
                
        # Create dataframe from dictionary
        data = pd.DataFrame.from_dict(all_words)
        data[" "] = ["spam","ham"]
        data = data.set_index(" ")

        N_spam, N_ham = data.sum(axis = 1).values

        for sms in X.values:
            split = sms.split()
            for word in split:
                n_spam = data[word]["spam"]
                n_ham = data[word]["ham"]
                spam_rates[word] = (n_spam + 1) / (N_spam + 2)
                ham_rates[word] = (n_ham + 1) / (N_ham + 2)
        
        self.data = data
        self.spam_rates = spam_rates
        self.ham_rates = ham_rates

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''

        num_spam_words, num_ham_words = self.data.sum(axis = 1).values

        P_spam = num_spam_words / (num_spam_words + num_ham_words) # 7.3
        P_ham = num_ham_words / (num_spam_words + num_ham_words) # 7.4

        spam_list = []
        ham_list = []

        for sms in X.values:
            split = sms.split()

            done = set()
            ham_sum = 0
            spam_sum = 0

            for word in set(split):
                # Don't want to double count words
                if word not in done:
                    done.add(word)
                    n = split.count(word)
                    len_message = len(split)

                    if word in self.data.loc["ham"]:
                        r_ham = self.ham_rates[word]
                        P_ham_word = (r_ham*len_message)**n * (np.exp(-r_ham*len_message)) /\
                            np.math.factorial(n)
                    else: 
                        P_ham_word = 1

                    if word in self.data.loc["spam"]:

                        r_spam = self.spam_rates[word]
                        P_spam_word = (r_spam*len_message)**n * (np.exp(-r_spam*len_message)) /\
                            np.math.factorial(n)
                    else:
                        P_spam_word = 1

                    ham_sum += np.log(P_ham_word)
                    spam_sum += np.log(P_spam_word)
                
            ham_list.append(ham_sum + np.log(P_ham))
            spam_list.append(spam_sum + np.log(P_spam))
        return np.vstack((ham_list, spam_list)).T

    def predict(self, X):
        '''
        Use self.predict_log_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''

        # Call predict_log_proba function
        predictions = self.predict_log_proba(X)

        # Get argmaxes and return array of ham for index 0 and spam for index 1
        maxs = np.argmax(predictions, axis = 1)
        return np.array(["ham" if g == 0 else "spam" for g in maxs])
        
        
def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''

    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)

    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)
    
    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts)

    return labels




# --- Testing code ---
# df = pd.read_csv("sms_spam_collection.csv")

# X = df.Message
# y = df.Label

# NB = NaiveBayesFilter()
# NB.fit(X[:300], y[:300])
# print(NB.predict_log_proba(X[530:535]))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)



# actual_labels = sklearn_method(X_train, y_train, X_test)
# # print(accuracy_score(actual_labels, y_test))

# NB = NaiveBayesFilter()
# NB.fit(X_train, y_train)
# NB_labels = NB.predict_log(X_test)
# print(accuracy_score(actual_labels, NB_labels))

# PB = PoissonBayesFilter()
# PB.fit(X_train, y_train)
# PB_labels = PB.predict(X_test)
# print(accuracy_score(actual_labels, PB_labels))






