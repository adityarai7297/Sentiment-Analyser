from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)

with open('cleaned_review_list.pkl', 'rb') as f:
    clean_train_reviews = pickle.load(f)


vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 


train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()






with open('cleaned_test_review_list.pkl', 'rb') as f:
    clean_test_reviews = pickle.load(f)



test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


