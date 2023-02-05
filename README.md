# Hate Speech Detection

The model in this project is a Convolutional Neural Network used to detect hate speech for SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. The original paper for this task can be found here: https://aclanthology.org/S19-2007/

## Preprocessing
The data for this task consist of hate speech and non-hate speech tweets in English and Spanish. Both the original and preprocessed datasets can be found in their corresponding folders. 

__Disclaimer:__ As this is a task on Hate Speech, the dataset contains many instances of this content, including racism, xenophobia and misogyny. 

A number of preprocessing steps were taken to help improve the performance of the model: 

- text cleaned using ftfy
- whitespace normalisation
- usernames were replaced by USER
- numbers replaced by NUM
- hashtags were normalised
- emojis were "translated" 
- spellcheck performed
- stopword removal
- contractions normalised
- slang abbreviations ("lol", "lmao", "tbh") normalised
- lemmatisation
- lowercasing

## Features
Three handcrafted features were added to the text data:

1. Sentiment score
2. Occurrences of all-caps words 
3. Occurrences of terms found in the Hatebase database (https://hatebase.org/)

All features were normalised to a range between 0 and 1.

## Model

![diagram_model2](https://user-images.githubusercontent.com/76164630/216852293-b5571b03-0b01-421a-9b56-7a10ff8775e9.JPG)

- GloVe embeddings were used to convert text into numerical data
- four convolutional layers with four different kernel sizes
- each followed by a ReLU and max pooling layer 
- outputs of all pooling layers concatenated into a 1-dimensional tensor
- tensor fed into linear layer, which outputs result: 0 for non-hate speech, 1 for hate speech
