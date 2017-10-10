import time
from csv import DictReader, DictWriter
import numpy as np
from numpy import array
from scipy.sparse import hstack as hst
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
# AA CHANGE: Added for debugging and making features
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn


kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kPAGE_FIELD = 'page'
kTROPE_FIELD = 'trope'


# IMDB Data Processing
kTITLE_FIELD = 'title'
kGENRE_FIELD = 'genre'


global_dead_list = ["dead","death","die","dies","died","finale", "show","turns","s","season","end","back","revealed"]
global_kill_list = ["kill","kills","killed","killing", "turns out", "the end", "the show", "end of", "the season", "out that", "one episode", "at the", "he s", "to kill","that he","in the"]
vocab = list(set(global_kill_list)|set(global_dead_list))

imdb_dict = list(DictReader(open("imdb_dict.csv", 'r')))
title_vec = []
genre_vec = []
for ii in imdb_dict:
    title_vec.append(ii[kTITLE_FIELD])
    genre_vec.append(ii[kGENRE_FIELD])


imdb_dict = dict(zip(title_vec, genre_vec))



class Featurizer:
    def __init__(self):
        # The precious vectorizer
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1,8),max_features=15000)  # analyzer='word' 'char' 'char_wb', strip_accents = 'ascii', ngram_range=(1,8), stop_words = 'english', max_df = 0.999, min_df = 0.0001
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)
    def test_feature(self, examples):
        return self.vectorizer.transform(examples)
    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-20:]
            bottom10 = np.argsort(classifier.coef_[0])[:20]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-20:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))








if __name__ == "__main__":
    start_time = time.time()
    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))



# -----------------------------------------USER WRITTEN CODE: CREATE VALIDATION SET------------------------------------------------

    # Creation of validation set
    stop = int((8 / 10) * len(train))
    start = stop + 1
    train_set = train[:stop]
    validation_set = train[start:]

# ------------------------------------------------------------------------------------------------------------


    feat = Featurizer()
    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))



# -----------------------------------------USER WRITTEN CODE: PERFORM TRAINING------------------------------------------------

    # Training the training data: Train using Sentence, Trope, and IMDb Genre
    x_train = feat.train_feature(str(x[kTEXT_FIELD]  + " " + x[kTROPE_FIELD] + " " + str(imdb_dict[x[kPAGE_FIELD]])) for x in train_set)
    # Normalizing the feature matrix
    x_train = normalize(x_train, axis=1, norm='l2')
    # Training the validation set
    val_train = feat.test_feature(str(x[kTEXT_FIELD]   + " " + x[kTROPE_FIELD] + " " + str(imdb_dict[x[kPAGE_FIELD]])) for x in validation_set)
    # Training the test set
    x_test = feat.test_feature(str(x[kTEXT_FIELD]   + " " + x[kTROPE_FIELD] + " " + str(imdb_dict[x[kPAGE_FIELD]])) for x in test)



    vect = TfidfVectorizer(vocabulary= vocab, max_features=5000, ngram_range=(1, 8))

    # Training the vocabulary
    x_train_voc = vect.fit_transform(str(x[kTEXT_FIELD]) for x in train_set)
    val_train_voc = vect.transform(str(x[kTEXT_FIELD]) for x in validation_set)
    x_test_voc = vect.transform(str(x[kTEXT_FIELD]) for x in test)


    # Concatenating the Features
    x_train = hst([x_train, x_train_voc]).toarray()
    val_train = hst([val_train, val_train_voc]).toarray()
    x_test = hst([x_test, x_test_voc, ]).toarray()


    # Collecting the labels for train and validation set for prediction purposes
    y_train = array(list(labels.index(x[kTARGET_FIELD]) for x in train_set))
    y_validation = array(list(labels.index(x[kTARGET_FIELD]) for x in validation_set))

# ------------------------------------------------------------------------------------------------------------



    print(len(train), len(test))
    print(set(y_train))
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)




# -----------------------------------------USER WRITTEN CODE: PRINT ACCURACY------------------------------------------------

    # User written code for printing out the training and validation set accuracy
    print(np.mean(y_train == lr.predict(x_train)))
    print(accuracy_score(y_validation, lr.predict(val_train)))

# ------------------------------------------------------------------------------------------------------------



    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
