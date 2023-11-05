import numpy as np

class Node:
    # helper class
    
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value
        
class DeciscionTree:
    
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _entropy(s):

        count = np.bincount(s)
        probabilities = count/len(s)
        
        for probability in probabilities:
            if probability > 0:
                entropy += probability*np.log2(probability)
        return - entropy


    def _information_gain(self, parent,left_child,right_child):
        left_num = len(left_child) / len(parent)
        right_num = len(right_child) / len(parent)
        
        gain = self._entropy(parent) - (left_num * self._entropy(left_child) + right_num * self._entropy(right_child))
        return gain

    def best_split(self, X, y):
        # calculates best split for for given features and target
        # where X is an array of features
        # y is an array of targets
        # returns a dictionary
        
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        for f_idx in range(n_cols):
            # X columns
            X_now = X[:,f_idx]
            # value of every unique feature
            for threshold in np.unique(X_now):
                # construct a data set split it into right and left
                # left includes record lower or equal to the threshold
                # right includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])
                
                # if data is in left and right calculate info gain
                if len(df_left) > 0 and len(df_right) > 0:
                    # get targets
                    y = df[:,-1]
                    y_left = df_left[:,-1]
                    y_right = df_right[:,-1]
                    
                    # get information gain
                    # if the current split is better than the previous best
                    gain = self._information_gain(y,y_left,y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
                        
        return best_split

                
                



