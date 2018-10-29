from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import pickle
import pandas as pd

with open('xtest.pkl', 'rb') as f1:
   Xtest = pickle.load(f1)

model = load_model('trained_model.h5')

Ytest_sentiment = model.predict_classes(Xtest)
#to flatten the list from [[0],[1],[0]....] to [0,1,0....]
Ytest_sentiment =Ytest_sentiment.flatten()
Ytest_sentiment =Ytest_sentiment.tolist()


test = pd.read_csv("testData.tsv", header=0, \
                    delimiter="\t", quoting=3)

Ytest_id = test['id'].tolist()
Ytest_id = [x.strip('"') for x in Ytest_id]

final_test_output = list(zip(Ytest_id, Ytest_sentiment))



#writing the results to a csv file

import csv



with open('final_test_output.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['id','sentiment'])
    for row in final_test_output:
        csv_out.writerow(row)