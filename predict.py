from build import build

def predict(self, x, tree):
    
    # recursively traverses tree to classify x
    
    # leaf node base case
    if tree.value !=None:
        return tree.value
    feature_value = x[tree.feature]
    
    # go to left
    if feature_value <= tree.threshold:
        return self.predict(x=x, tree=tree.data_left)
    
    # go to right
    if feature_value > tree.threshold:
        return self.predict(x=x, tree=tree.data_right)
    
    