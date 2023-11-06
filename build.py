from best_split import best_split
from node import Node
from collections import Counter

def build(self, X, y, depth=0):
    # builds decision tree
    
    # X feature vector
    # y target vector
    # current depth of tree (base case)
    # returns node
    
    n_rows, n_cols = X.shape
    
    # check if node should be a leaf
    if n_rows >= self.min_samples_split and depth <= self.max_depth:
        # get best split
        best = self.best_split(X,y)
        # if the split is bad
        if best['gain'] > 0:
            # build left side of tree
            left = self.build(
                X=best['df_left'][:,:-1],
                y=best['df_left'][:,:-1],
                depth=depth + 1
            )
            # build right side of tree
            right = self.build(
                X=best['df_right'][:,:-1],
                y=best['df_right'][:,:-1],
                depth=depth + 1
            )
            return Node(
                feature=best['feature_index'],
                threshold=best['threshold'],
                data_left=left,
                data_right=right,
                gain=best['gain']
            )
    return Node(
        value=Counter(y).most_common(1)[0][0]
    )
            