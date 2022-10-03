# Sentiment Analysis
Given a movie review, use machine learning model to extract subjective information, i.e., determine whether it is positive or negative.

## Workflow
1. [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) is used for training and validation
2. Tokenization
3. Preprocessing
    1. Eliminate numbers and punctuation marks
    2. Lemmatization
    3. Negation
    4. Eliminate stop words
    5. Convert all characters to lowercase
4. Vectorization: 
    - Combine TFIDF and Word Embedding, take the mean of (TFIDF × Word Embedding)
    - Word Embedding either uses the model training by IMDB dataset or pretrained Glove, and experiment with word vector dimension to be 100 or 200
5. Input the processed data into a binary classification model: 
    - Experiment with Naive Bayes (NB), Support Vector Machine (SVM), and Deep Neural Network
6. Use 5-fold cross validation to evaluate models' performance, including accuracy, precision and recall
7. Inference


## Results
### Cross Validation
|          Model                        | Accuracy | Precision | Recall |
| ------------------------------------- | -------- | --------- | ------ |
| dim(wv) = 100, W2V + NB               |   0.752  |   0.749   |  0.753 |
| dim(wv) = 100, W2V + SVM              |   0.875  |   0.867   |  0.882 |
| dim(wv) = 100, W2V + DNN              |   0.866  |   0.858   |  0.879 |
| dim(wv) = 100, Pretrained Glove + NB  |   0.719  |   0.728   |  0.716 |
| dim(wv) = 100, Pretrained Glove + SVM |   0.814  |   0.807   |  0.818 |
| dim(wv) = 200, W2V + NB               |   0.712  |   0.723   |  0.707 |
| dim(wv) = 200, W2V + SVM              |   0.876  |   0.868   |  0.883 |
| dim(wv) = 200, W2V + DNN              |   0.875  |   0.867   |  0.882 |
| dim(wv) = 200, Pretrained Glove + NB  |   0.734  |   0.742   |  0.735 |
| dim(wv) = 200, Pretrained Glove + SVM |   0.837  |   0.824   |  0.845 |

Note: dim(wv) means word vector dimension.

Observing from the table, using IMDB dataset trained Word2Vec model and TFIDF to vectorize words to 200-dimensional word vector, and then input the processed data into SVM binary classification model leads to the best result.

### Inference
Testing with the best model selected from cross validation gave rise to the following results.

| Input | Output |
| ----- | ------ | 
| "This Spiderman is really fantastic. It captivated me right from the start, and I was entranced every second." | positive |
| "Another worst action film! Full of annoying overuse scene, and not intense action scene! Not recommended!" | negative |
| "Encanto is a creative movie featuring beautiful and vibrant animation. However, the story feels a little underdeveloped. While there are some magical and emotional moments, it seems as if they didn't know how to end the movie." | positive |


## Directory
```
/
├── w2v-100.ipynb          # Training with word vector dimension = 100
├── w2v-200.ipynb          # Training with word vector dimension = 200
└── best (w2v+SVM).ipynb   # Inference by the the best model
```
