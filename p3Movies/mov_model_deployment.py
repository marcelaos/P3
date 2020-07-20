#!/usr/bin/python

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

import sys
import os
import numpy as np

def predict_proba(plot):

    clf = joblib.load(os.path.dirname(__file__) + '/model_movie_clf.pkl') 
    dataTraining = pd.read_csv('https://github.com/albahnsen/AdvancedMethodsDataAnalysisClass/raw/master/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)


    df2 = pd.DataFrame (columns = ['year','title', 'plot'])
    df2 = df2.append({'year':'1985','title':'title', 'plot':plot}, ignore_index=True)
    vect = CountVectorizer(max_features=1000)
    vect.fit_transform(dataTraining['plot'])
    
    document = [' '.join(str(item)) for item in df2['plot']]
    X_test_dtm2 = vect.transform(document)

    #X_test_dtm2 = vect.transform(df2['plot'])
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
            'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
            'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    y_pred_test_genres2 = clf.predict_proba(X_test_dtm2)

    p1 = pd.DataFrame(y_pred_test_genres2, columns=cols)

    
    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a PLOT')
        
    else:

        plot = sys.argv[1]

        p1 = predict_proba(plot)
        
        print(plot)
        print('Probability of genres: ', p1)
        