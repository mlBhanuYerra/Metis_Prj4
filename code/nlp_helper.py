import numpy as np
import pandas as pd
import re, string
import os

#from contractions_re import * #for expandContractions(text)
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from langdetect import detect
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from joblib import dump, load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


# --- Stopwords
PRINTABLE = [i for i in string.printable]
GENREIC = ["ain't", "aren't", "can't", "can't've", "'cause", "could've", "couldn't", "couldn't've", "didn't", "doesn't", "don't", "hadn't", "hadn't've", "hasn't", "haven't", "he'd", "he'd've", "he'll", "he'll've", "he's", "how'd", "how'd'y", "how'll", "how's", "i'd", "i'd've", "i'll", "i'll've", "i'm", "i've", "isn't", "it'd", "it'd've", "it'll", "it'll've", "it's", "let's", "ma'am", "mayn't", "might've", "mightn't", "mightn't've", "must've", "mustn't", "mustn't've", "needn't", "needn't've", "o'clock", "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've", "she'd", "she'd've", "she'll", "she'll've", "she's", "should've", "shouldn't", "shouldn't've", "so've", "so's", "that'd", "that'd've", "that's", "there'd", "there'd've", "there's", "they'd", "they'd've", "they'll", "they'll've", "they're", "they've", "to've", "wasn't", "we'd", "we'd've", "we'll", "we'll've", "we're", "we've", "weren't", "what'll", "what'll've", "what're", "what's", "what've", "when's", "when've", "where'd", "where's", "where've", "who'll", "who'll've", "who's", "who've", "why's", "why've", "will've", "won't", "won't've", "would've", "wouldn't", "wouldn't've", "y'all", "y'all'd", "y'all'd've", "y'all're", "y'all've", "you'd", "you'd've", "you'll", "you'll've", "you're", "you've", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
SUBJECT_RELATED = ['cookbook', 'cookbooks', 'book', 'books', 'recipes', 'like', 'copyright', 'information', 'cook', 'home', 'best', 'food', 'vegetarian']
MISC_STOPWORDS = ['com', 'xa']
ADD_STOPWORDS = ["'caus", "'d", "'ll", "'m", "'re", "'s", "'ve", 'abov', 'afterward', 'ai', 'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 'ca', 'cri', 'describ', 'did', 'doe', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifi', 'formerli', 'forti', 'ha', 'henc', 'hereaft', 'herebi', 'hi', 'howev', 'hundr', 'inde', 'inform', 'latterli', 'let', 'mani', 'meanwhil', 'moreov', 'mostli', "n't", 'need', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ought', 'ourselv', 'perhap', 'pleas', 'recip', 'seriou', 'sever', 'sha', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv', 'twenti', 'veri', 'wa', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'wo', 'yourselv']
COOKBOOK_STOPWORDS = PRINTABLE + GENREIC + SUBJECT_RELATED + MISC_STOPWORDS + ADD_STOPWORDS

cookbook_mwe = MWETokenizer([
        ('low', 'carb'),
        ('fat', 'loss'),
        ('fat', 'free'),
        ('gluten', 'free'),
        ('sugar', 'free'),
        ('low', 'fat'),
        ('meat', 'less'),
        ('instant', 'pot'),
        ('slow', 'cooker'),
        ('mini', 'bar'),
        ('dairy', 'free'),
        ('for', 'one'),
        ('cast', 'iron'),
        ('new', 'york'),
        ('san', 'francisco'),
        ('los', 'angeles'),
        ('betty', 'crocker')
])

VECTORIZATION = {
    'count': CountVectorizer,
    'tfidf': TfidfVectorizer
}

DIMENSIONALITY_REDUCTION = {
    'PCA': PCA,
    'LSA': TruncatedSVD,
    'NMF': NMF,
    'LDA': LatentDirichletAllocation
}

CLUSTERING = {
    'kmeans': KMeans
}



##########################################################################
# --- Helper Classes
class nlp_model:
    """
    
    #--- Parameters and thier defaults if applicable:
    vect_method = "count"
    dim_reduc="LSA"
    clustering="kmeans"
    n_components = 21
    n_clusters=6
                           
    #--- Attributes:
    self.dataNLP
    self.methods = A dictionary of strings representing pipeline of methods used for NLP model
        example: self.methods is {'Vectorization': 'count',
                                  'Dimensionality Reduction': 'LSA',
                                  'Clustering': 'kmeans'
                                  }                             
    self.objects = A dictionary of objects representing instances of methods used for NLP model
        created as part of the model fit self.fit() method
    self.outputs = A dictionary of primary outputs of pipeline used for NLP model
        created as part of the model fit self.fit() method
    self.n_components = number of components for dimensionality reduction
    self.n_clusters = number of clusters for clustering algorithm (might change with other
                      clustering algorithm inclusions in the future)
    
    #--- Methods:
    
    """
    
    def __init__(self, vect_method = "count", dim_reduc="LSA", clustering="kmeans",
                 n_components = 21, n_clusters=6):
        """
        """
        #self.doc_text = data
        
        ##########################################################################
        #--- Create self.methods attribute from the class parameters
        self.methods = {}
        
        # --- Token Vectorization Methods
        if vect_method == "tfidf":
            self.methods['Vectorization'] = vect_method
        else:
            self.methods['Vectorization'] = "count"
            
        # --- Dimensionality Reduction Methods
        if dim_reduc == "PCA" or dim_reduc == "NMF" or dim_reduc == "LDA":
            self.methods['Dimensionality Reduction'] = dim_reduc
        else:
            self.methods['Dimensionality Reduction'] = "LSA"
        
        # --- Clustering Methods (Only kmeans at the moment. Expand based on need)
        if clustering == "kmeans":
            self.methods['Clustering'] = clustering
        else:
            self.methods['Clustering'] = "kmeans"
        
        ##########################################################################
        # --- number of components for dimensionality reduction
        self.n_components = n_components
        
        # --- number of clusters for clustering
        self.n_clusters = n_clusters
        
    def fit(self, data, save_fit = False, filename = None):
        """
        Function to fit the model using the data from the arguments
        Runs through the pipeline as defined in the object instantiation
        
        Argutments:
        data - nlp data in docs X 1 format, and should be readable by vectorization method
        
        Returns: None object
        """
        ##########################################################################
        # --- Get temp methods, once instantiated will be saved under self.objects
        to_vect = VECTORIZATION[self.methods['Vectorization']]
        to_dim_reduce = DIMENSIONALITY_REDUCTION[self.methods['Dimensionality Reduction']]
        to_cluster = CLUSTERING[self.methods['Clustering']]
        
        ##########################################################################
        self.dataNLP = data
        
        self.objects = {}
        self.outputs = {}
        
        #--- Vectorization step
        self.objects['Vectorization'] =  to_vect(stop_words="english")
        vect_data = self.objects['Vectorization'].fit_transform(self.dataNLP)
        self.outputs['Vectorization'] = vect_data.toarray()
        
        #--- Dimensionality Reduction step
        self.objects['Dimensionality Reduction'] = to_dim_reduce(n_components=self.n_components)
        self.outputs['Dimensionality Reduction'] = self.objects['Dimensionality Reduction'].fit_transform(self.outputs['Vectorization'])
        
        #--- Clustering step
        self.objects['Clustering'] = to_cluster(n_clusters=self.n_clusters, random_state=30)
        self.objects['Clustering'].fit(self.outputs['Dimensionality Reduction'])
        self.outputs['Clustering'] = self.objects['Clustering'].labels_
        
        if save_fit and filename:
            dump(self, filename)
        
        
##########################################################################
# --- Helper Functions

# Use TextBlob
# credit: http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

# --- To plot n-dimensional data 
def plot_all_dimensions(data, class_labels, title=""):
    
    plt.rcParams['figure.figsize'] = [30, 20]
    
    dims = data.shape[1]
    assert dims <= 6
    
    count = 1
    for dim in range(dims):
        i = dim
        j = (i+1) * int(dim != dims-1)
        
        plt.subplot(2,3, count)
        plt.scatter(data[:,i], data[:,j],# ckbk_space[vect][dr][:,2],
                   s = 4, alpha = 0.5, c=class_labels)
        plt.title(title, fontsize=20, color='red')
        count += 1
    plt.show()


def plot_all_dimensions3D(data, class_labels, titles=""):
    
    plt.rcParams['figure.figsize'] = [30, 20]
    fig = plt.figure()
    
    dims = data.shape[1]
    assert dims <= 6
    
    count = 1
    for dim in range(dims):
        i = dim
        j = (i+1) * int(dim != dims-1)
        k = (j+1) * int(j != dims-1)
        
        ax = fig.add_subplot(2,3, count, projection='3d')
        ax.scatter(data[:,i], data[:,j], data[:,k],
                   s = 4, alpha = 0.5, c=class_labels)
        ax.set_title(titles, fontsize=20, color='red')
        ax.grid(False)
        count += 1
    plt.show()
    
# --- Parser for reading in the Amazon json files (can be used for both reviews and metadata)
# --- credits folloing parse() method to Julian McAuley UCSD: http://jmcauley.ucsd.edu/data/amazon/ 
def parse(path):
    g = open(path, 'r')
    for l in g:
        yield json.loads(l)
        

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize_col(pd_series):
    """
    """
    # --- Helpers
    alphanumeric = lambda x: re.sub('[\d]+', ' ', x)
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    remove_stopwords = lambda x: " ".join([word for word in x if word not in COOKBOOK_STOPWORDS])
    wnl = WordNetLemmatizer()
    
    col = pd_series.name
    newDF = pd.DataFrame(pd_series)
    newDF['text_step1'] = newDF[col].apply(lambda x: expandContractions(x.lower()))
    newDF['text_step2'] = newDF['text_step1'].map(alphanumeric)
    newDF['text_step3'] = newDF['text_step2'].map(punc_lower)
    newDF['text_step4'] = newDF['text_step3'].apply(lambda x: " ".join(cookbook_mwe.tokenize(word_tokenize(x))))
    newDF['text_step5'] = newDF['text_step4'].apply(lambda x: pos_tag(word_tokenize(x)))
    newDF['text_step6'] = newDF['text_step5'].apply(lambda x: [(word, get_wordnet_pos(pos)) for word, pos in x])
    newDF['text_step7'] = newDF['text_step6'].apply(lambda x: [wnl.lemmatize(word, pos) for word, pos in x])
    newDF['text_step8'] = newDF['text_step7'].map(remove_stopwords)
    
    return newDF['text_step8']

# ---
# Credit for the following code: https://devpost.com/software/contraction-expander
#

import re
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

# examples
#print(expandContractions("Don\'t you get it?".lower()))
#print(expandContractions('I ain\'t got time for y\'alls foolishness'))
#print(expandContractions('You won\'t live to see tomorrow.'.lower()))
#print(expandContractions('You\'ve got serious cojones coming in here like that.'.lower()))
#print(expandContractions('I hadn\'t\'ve enough'))