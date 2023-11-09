from predict import predict

def new_predict(self, X):
    
    # classifies new instances
    
    # calls predict function to every observation
    
    return[self.predict(x, self.root) for x in X]