import numpy as np

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the number of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

class NaiveBayesClassifier(HateSpeechClassifier):

    def __init__(self):
        self.posarray = []
        self.negarray = []
        self.ypostotal = 0
        self.ynegtotal = 0
    

    def fit(self, X, Y):

        # initialize frequency arrays for words
        postemp = np.zeros(X[0].size)
        negtemp = np.zeros(X[0].size)
        poswords = 0
        negwords = 0

        for i in range(len(X)):

            #total number of negative or positive labels
            if(Y[i] == 0):
                self.ynegtotal += 1
            else:
                self.ypostotal += 1
                
           # go through each word
            for x in range(len(X[i])):
       
                # if negative label
                if(Y[i] == 0):
                    
                   #add one occurence of the word as negative
                    negtemp[x] += X[i][x]
                    negwords += X[i][x]
                   

                    # add one occurence of the word as positive
                else:
                    postemp[x] += X[i][x]
                    poswords += X[i][x]

        # add laplace smoothing here if word not found lam = + 1.1 and V = len(X[0]) / otherwise find the probability of pos or neg compared to word count for pos and neg
        for i in range(len(postemp)):
            postemp[i] = max((postemp[i]/poswords), ((postemp[i]+1.1)/(poswords + len(postemp))) )

        for i in range(len(negtemp)):
            negtemp[i] = max((negtemp[i]/negwords), ((negtemp[i]+1.1)/(negwords + len(postemp))))
            
    

        # set the arrays and for the total values divide by total number of entries size
        self.posarray = postemp
        self.negarray = negtemp
        self.ypostotal /= X[0].size
        self.ynegtotal /= X[0].size


    def predict(self, X):

        # initialize return array
        result = []

        # for each sentence
        for i in range(len(X)):


            # add the probability of pos or neg label with the sum of all the probabilities of a word being positive/negative given the word
            postotal = np.log(self.ypostotal) +  np.sum((np.log(self.posarray) * X[i]))
            negtotal =  np.log(self.ynegtotal) +  np.sum((np.log(self.negarray) * X[i]))

            
            # don't need to divide the probs with the same sum and we can just compare our two predicted values for positive or negative label here
            if postotal > negtotal:
                result.append(1)
            else:
                result.append(0)
       
        
        return result


class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, lr=0.1, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        
        # initialize parameters
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        
        # gradient descent
        for i in range(self.epochs):
            # compute linear combination and sigmoid activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y)
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred > 0.5).astype(int)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

class L2RegLogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, lr=0.01, epochs=5000, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        
        # initialize parameters
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        
        # gradient descent
        for i in range(self.epochs):
            # compute linear combination and sigmoid activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # compute gradients with L2 regularization
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - Y)) + self.lam * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - Y)
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred > 0.5).astype(int)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))




