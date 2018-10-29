import pandas as pd  
from bs4 import BeautifulSoup 
import re   
from nltk.corpus import stopwords 
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
letters_only= ""
cleaned_sentence=[]
cleaned_reviews=[]
count=0
for i in range(25000):
    example1 = BeautifulSoup(train["review"][i],features="lxml")  
  
    print("reached review number ",count," out of 25000 reviews")

    letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text() )          
    lower_case = letters_only.lower()        
    words = lower_case.split()                              
    for w in words:
       
        if not w in stopwords.words("english"):
            cleaned_sentence.append(w)
        sentence = " ".join(cleaned_sentence)
    cleaned_reviews.append(sentence)
    cleaned_sentence=[]
    sentence=""
    count=count+1
                    


#words = [w for w in words if not w in stopwords.words("english")]


import pickle

with open('cleaned_review_list.pkl', 'wb') as f:
    pickle.dump(cleaned_reviews, f)


print(cleaned_reviews[3])     


