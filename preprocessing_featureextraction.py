import pandas as pd
import re
import nltk
import spacy
import csv
import re
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from contractions import contractions as con          # file for mapping contractions to full forms
import string
import emoji
import ftfy
from spellchecker import SpellChecker
import wordsegment
from wordsegment import load, segment
from abcs import hashtags as hs, caplist, slang       # file for normalising hashtags, detecting all-caps words and "translating" slang abbreviations
load()
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis_spanish import sentiment_analysis

#dtf = pd.read_csv("en_train.csv", delimiter=",", na_filter=False, encoding="utf-8")
#dtf = pd.read_csv("en_test.csv", delimiter=",", na_filter=False, encoding="utf-8")
dtf = pd.read_csv("en_dev.csv", delimiter=",", na_filter=False, encoding="utf-8")

#spanish version
#dtf = pd.read_csv("es_train.csv", delimiter=",", na_filter=False, encoding="utf-8")
#dtf = pd.read_csv("es_test.csv", delimiter=",", na_filter=False, encoding="utf-8")

hatebase = pd.read_csv("updated_hatebase_english.csv", delimiter=",", na_filter=False, encoding="utf-8")
#hatebase = pd.read_csv("hatebase_vocab_spanish.csv", delimiter=",", na_filter=False, encoding="utf-8") #spanish

stop = stopwords.words('english')
#stop = stopwords.words("spanish")          #spanish version

nlp = spacy.load("en_core_web_md")
#nlp = spacy.load("es_core_news_md")         #spanish version

sia = SentimentIntensityAnalyzer()
#sentiment = sentiment_analysis.SentimentAnalysisSpanish()   #spanish

spell = SpellChecker()
#spell = SpellChecker(language = "es")      #spanish version


hashlist = hs.items()
listcon = con.items()
slanglist = slang.items()


def lemmatise(tweet):
    doc = nlp(tweet)
    result = [token.lemma_ for token in doc]    #lemmatization
    tweet = " ".join(result)
    return tweet

def removal(tweet):
    tweet = re.sub(r"http\S+", "", tweet)       #remove URLS
    #tweet = re.sub(r"\S*@\S*\s?", "", tweet)    #remove emails
    #tweet = tweet.translate(str.maketrans('', '', string.punctuation))  #remove punctuation, most non-alphanumerical chars
    return tweet

def demoji(tweet):
    tweet = (emoji.demojize(tweet))
    #tweet = (emoji.demojize(tweet, language = "es"))   #spanish version
    return tweet

def remove_stop(tweet):
    tweet_tok = word_tokenize(tweet)
    my_list = [word for word in tweet_tok if word not in stop]
    stoppito = " ".join(my_list)
    return stoppito

def replace(tweet):
    tweet = re.sub("@[A-Za-z0-9_]+","user", tweet)      #replace mentions with USER
    tweet = re.sub(r"\d+", "num", str(tweet))           #Replaces digits with 'NUM'
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)           #replaces repeating chars w 2 occurrences
    return tweet

def contractions(tweet):    #normalises contractions, e.g. I've -> I have
    for key, value in listcon:
        tweet = tweet.replace(key, value)
    return tweet

def fixmalformed(tweet):
    tweet = ftfy.fix_text(tweet)
    tweet = re.sub("&;","and", tweet)
    return tweet

def spellcheck(tweet):
    tweet = tweet.split()
    tweet = [spell.correction(x) for x in tweet]
    return " ".join(tweet)

def spellcheck_es(tweet):
    result = []
    tweet = tweet.split()
    for x in tweet:
        if x != "user":
            result.append(spell.correction(x))
        else:
            result.append(x)
    return " ".join(result)

def flag_all_caps(tweet):
    caps = re.findall('([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)', tweet)
    caps = " ".join(caps)
    for key in caplist:                                #no caplist for spanish
        caps = caps.replace(key, "")
    caps = " ".join(caps.split()) 
    doc = nlp(caps)
    counter = 0
    for token in doc:
        counter = counter + 1
    return counter

def normalise_whitespace(tweet):
    tweet = " ".join(tweet.split())
    return tweet
    
def hashtagsegmentation(tweet):
    for key, value in hashlist:                        #no hashlist for spanish
        tweet = tweet.replace(key, value)
    hashtags = re.findall(r"(#\w+)", tweet)
    for x in hashtags:
        temp = " ".join(segment(x))
        tweet = tweet.replace(x, temp)
    return tweet

def sentimentanalysis(tweet):
    return(sia.polarity_scores(tweet)["compound"])
    #return(sentiment.sentiment(tweet))                  #spanish version

def punctuation(tweet):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet

def separate_words(text):
    text = text.split()
    A = text[:len(text)//2]
    A = " ".join(A)
    B = text[len(text)//2:]
    B = " ".join(B)
    a = (segment(A))
    b = (segment(B))
    tweet = a+b
    tweet = " ".join(tweet)
    return tweet

def replace_slang(tweet):
    for key, value in slanglist:
        tweet = tweet.replace(key, value)
    return tweet

def check_hatebase(tweet):
    counter = 0
    for i in range(len(hatebase)):
        if hatebase.loc[i, "term"] in tweet:
            counter = counter+1
    return counter

dtf["text"] = dtf["text"].apply(fixmalformed)
dtf["text"] = dtf["text"].apply(removal)
dtf["text"] = dtf["text"].apply(replace)
dtf["caps"] = dtf["text"].apply(flag_all_caps)
dtf["text"] = dtf["text"].apply(hashtagsegmentation)              #no available segmentation package for spanish
dtf["text"] = dtf["text"].apply(demoji)
dtf["text"] = dtf["text"].apply(normalise_whitespace)

dtf["sntmt"] = dtf["text"].apply(sentimentanalysis)
dtf["text"] = dtf["text"].apply(lambda x: x.lower())
dtf["text"] = dtf["text"].apply(replace_slang)                     #no slang database for spanish
dtf["text"] = dtf["text"].apply(contractions)
dtf["text"] = dtf["text"].apply(spellcheck)
#dtf["text"] = dtf["text"].apply(spellcheck_es)                      #spanish spellcheck version

#dtf["text"] = dtf["text"].apply(separate_words)
dtf["text"] = dtf["text"].apply(lemmatise)
dtf["text"] = dtf["text"].apply(remove_stop)
dtf["text"] = dtf["text"].apply(punctuation)
dtf["text"] = dtf["text"].apply(normalise_whitespace)
dtf["nohs"] = dtf["text"].apply(check_hatebase)

dtf.to_csv("preprocessed_en_train.csv", index=False, encoding="utf-8", sep=",")     #change name accordingly
