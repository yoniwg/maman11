class AbstractSolution3:
    
    ''' 
    The class template to inherit for your submission. 
    
    You should just stuff your code into the appropriate methods speced below. 
    See the provided example implementation for further inspiration, the provided example
    implementation simply stuffs the necessary stuff from the staff published notebook,
    into this interface.
    
    You do not need to implement a prediction/inference function for your submission, 
    as the submission scoring code performs prediction/inference from your model.
    '''
        
    def __init__(self):
        pass # no-op
        
    def vectorize(self, token):
        ''' 
        Your function turning a token into a features vector according to your feature scheme.
        The submission scoring code will invoke this function to vectorize data before passing it on
        to your `predict` function. 
        
        So it's a good idea to use this function in your training code as well
        
        Args:
            token: an input token
        
        Returns: 
            a vector
        '''

    def train(self, tokens, y):
        ''' 
        Your function training your model using the given tokens and their corresponding true labels y.
        This IS the training set to use for the training, you do not need to split it, you need to 
        train on ALL OF IT.
         
        Args:
            tokens: a list of tokens.
            y:      a list of true labels, corresponding in order to the list of tokens
           
        Returns: 
            a fitted sklearn model, either a MultinomialNB, or a LogisticRegression one
        '''
                
