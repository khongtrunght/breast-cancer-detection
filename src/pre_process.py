import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

def pre_process():
    def read_data(filename):
        data = pd.read_csv(filename, header= None, names = ['assessment','age','shape','margin','density','severity'], na_values= '?')
        data = data.dropna()

        dmmargin = pd.get_dummies(data['margin'], prefix= 'margin')
        dmshape = pd.get_dummies(data['shape'], prefix= 'shape')
        data.drop('shape', axis= 1, inplace=True)
        data.drop('margin', axis= 1, inplace=True)
        data = data.join(dmshape)
        data = data.join(dmmargin)
        cols = data.columns.tolist()
        cols.remove('severity')
        cols.append('severity')
        data = data[cols]
        # positive = data.loc[data.severity.eq(1)]
        # negative = data.loc[data.severity.eq(0)]
        # data.loc[data.severity.eq(1)] = data.fillna(positive.mean())
        # data.loc[data.severity.eq(0)] = data.fillna(negative.mean())
        X = data.iloc[:,1:-1]
        Y = data.iloc[:,-1]
        return X, Y
    
    def normalize_and_add_one(X):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X
    
    filename = os.path.join(os.path.dirname(__file__),"../data/project.data")
    X, Y = read_data(filename)
    X = normalize_and_add_one(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)    

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = pre_process()
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)

    print(cm)

    print((cm[0,0] + cm[1,1]) / np.sum(cm))