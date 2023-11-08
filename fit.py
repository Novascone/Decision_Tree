from build import build

# traverses tree to classify a single instance

def fit(self, X, y):
    # trains decision tree
    # X array of features
    # y array of targets
    
    self.root = self.build(X,y)
    
