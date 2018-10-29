# Sentiment-Analyser

I have used the bag of words encoding technique by using the sklearn library to vectorise the reviews to make it appropriate for machine learning.

I have used the keras library working on tensor flow backend to create a deep neural network model to predict the sentiment of movie reviews.

The neural network I used was a sequential neural network consisting of 2 hidden layers of 50 neurons each connected by the relu activation function which is connected to a final output layer consisting of 1 neuron on which a sigmoid activation is applied to give the result.

An overview of the program and data files included in the zip file:

labeledTrainData.tsv -
Contains 25000 movie reviews along with their sentiment values.(Rename your file accordingly)

testData.tsv -
Contains 25000 movie reviews which need to be assigned a sentiment value.(Rename your file accordingly)

cleaner.py - 
Used to clean the training reviews by removing html characters ,stop words, numbers and punctuations.

test_cleaner.py - 
Used to clean the testing reviews.

cleaned_review_list.pkl - 
Contains the cleaned training reviews which are then passed to the tokenizer.py for tokenization.

cleaned_test_review_list.pkl - 
Contains the cleaned test reviews which are then passed to the tokenizer.py for tokenization.

tokenizer.py - 
Used to vectorise the test and train reviews based on the bag of words model.

xtrain.pkl -
Contains the vectorised train reviews which which I got as a result from the tokenizer.py script. It is used to train the model.

xtest.pkl -
Contains the vectorised test reviews which which I got as a result from the tokenizer.py script. It is passed to the “predict.py” script for evaluation of sentiment.

model.py -
Contains the actual model to which the xtrain and the ytrain ( sentiment column of the labeledTrainData.tsv file) arrays are passed for training.

trained_model.h5 - after training, I saved the trained model as “trained_model.h5” to load it at the time of prediction by the predict.py script.

predict.py -
This script loads the model from the “trained_model.h5” and predicts the test sentiment values and stores the result in “final_test_output.csv” file.

final_test_output.csv -
The final output file.


The following libraries need to be installed in the programming environment :

Tensorflow
Keras
Pickle
Pandas
NumPy
SKlearn
Nltk
BeautifulSoup


Execution :

labeledTrainData.tsv 
testData.tsv 
cleaner.py 
test_cleaner.py  
tokenizer.py 
model.py
predict.py

The above files should be present in your working directory and then cd to that directory

Order of execution :
python cleaner.py
python  test_cleaner.py  
python  tokenizer.py 
python  model.py
python  predict.py


I have also given the intermediate data files:

 Running the cleaner.py script will generate cleaned_review_list.pkl file
 Running the test_cleaner.py script will generate cleaned_test_review_list.pkl file
 Running the tokenizer.py script will generate xtrain.pkl and xtest.pkl file
 Running the model.py script will generate trained_model.h5 file
 Running the predict.py script will generate final_test_output.csv file which is the expected output

