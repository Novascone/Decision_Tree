import numpy as np
from information_gain import information_gain
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
                gain = self.information_gain(y,y_left,y_right)
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

            
            