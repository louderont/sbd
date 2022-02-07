# Sentence Boundary Detection project

## Installation

At first, create a dedicated environment and activate it.
```
conda create -n py38_sbd python=3.8
conda activate py38_sbd
pip install -r requirements.txt
```
One of the dependencies -xgboost- required OMP binaries installed:\
On MacOS:
```
brew install libomp
```

Then, install the package in developper mode to always work with the lastest version of the package and from the root of the package (same level than setup.py):
```
pip install -e .
```
The package is now available accross the environment.

The folder structure of the project is detailed in a section below.

### CLI interface

To use the model to detect sentences in a text:
- Put a .txt file in an inputpath directory from the package root (default: 'sbdetection/data/input/test_text.txt')
- Then run:
```
sbdetection --inputpath path_input.txt --outputpath path_output.txt --modelname ML --modelpath mymodel.pkl
```
*inputpath*: path of the input text from the sbdetection folder. (default: default='sbdetection/data/input/test_text.txt')\
*outputpath:* path of the output of the model from the sbdetection folder (default: default='sbdetection/data/output/sentences.txt')\
*modelname*: ML or baseline (rule based) models\
*modelpath*: filename of the pickle model used to predict. The **default path is model_rf.pkl (serialized model from the training).**

It returns in the output file a list of string, each string corresponding a detected sentences.

## Problematic

The goal of the project is to provide a model performing sentence boundary detection (SBD) that consists in splitting a natural language text into sentences. The provided model should take an input text and return a list of sentences.


I choose to reduce the scope to focus on SBD on structured text with complete, cleaned and formal sentences as the latter are likely to be in this format in books and articles and in order to reduce the scope of the problem for this short project. I also focus on english texts.


In english, the end of a formal sentence should be a full stop: '?', '!', '.'. The end punctuation depends on the type of the text (story book, article, financial documents ..). Less formal sentences from texting, social media or corresponding to grammatical erros may end with emoticons and or with no ponctuation at all.\
I will then focus on texts with sentences ending with a full stop (., !, ?). I do not consider colons (:) or semi-colons (;) to be an end of a sentence and exclude headlines from the input text.


The difficulty of the task within this scope is to distinguish a sentence ending punctuation mark from a mark that may be associated with an abbreviation, an initial, a decimal number, be part of a quote or between dashes ect.

### Corpus

The task highly depends on the chosen train set and on the selected metric and test sets to evaluate the model.


To cover the scope of cleaned and formal sentence tokenization, the chosen text samples are extracted from novels and articles. As the first approach is chosen supervised, the selected train and test sets are extracts from already existing and tagged (sentence tokenized) corpus that are easily accessible using nltk and provide important volumes of data. 


To evaluate the robustness of the model on texts whoose sources are different from the train set's and with different rate of sentence end among full stop punctuations, I choose to use 3 different corpus.


As several papers on SBD used the Brown and the Treebank corpus as  datasets of cleaned and formal english sentences, I also select them as well as a book extract from the Gutenberg corpus:
- The Brown Corpus: a large and various english corpus created in 1961 at Brown University gathering texts from 500 sources that are classified by gender (news, editorial, fiction, news).
- The treebank corpus: 10% sample of the Penn Treebank corpus gathering Wall Street Journal samples.
- The Gutenberg Corpus containing 25 000 free e-books and whoose project is hosted here: http://www.gutenberg.org/.
\
The text sources are accessible using nltk: https://www.nltk.org/book/ch02.html 


A descriptive analysis of the datasets is provided in the next sections.

## Modeling

### Approach 

To provide an element of comparison, a basic rule based model is at first implemented using RegEx (Regular Expression).

The proposed approach is a token-classification model: the idea is to cut the text into word tokens and then to associate a binary target to each token (1: sentence end token, 0: not an end sentence token) to train a supervised classification model. I choose this approach as it is a straightforward way to address the issue that allows to perform feature extraction from the tokens and then to apply common and already implemented classification estimators using scikit-learn.
This token approach is also used in papers obtaining high scores on the targetted types of text (Treebank or Brown corpus) such as:
Sentence Boundary Detection Using a MaxEnt Classifier
Nishant Agarwal, K. Ford, M. Shneider, 2005

### Word tokenization

The *word_tokenize* function of nltk is avoided as it already includes SBD and a quick blank space separator based on regex is used instead.
Punctuation is thus included in the tokens to go for a quicker and less misleading approach to implement to rebuild the sentences at the end of the prediction going from tokens to sentences.

### Descriptive Analysis of the modeling sample sets

| Corpus source | Brown corpus | Treebank corpus (WSJ) | Gutenberg corpus |
| --- | --- | --- | --- |
| Nb of sentences | 3000 | 3000 | 30 000 |
| Avg nb of tokens per sentence | 73 | 135 | 136 |
| Part of ending tokens among all tokens | 7.5% | 4.4% | 3.9% |
| Part of ending tokens among tokens with full stops | 0.98 | 0.65 | 0.88 |


The part of ending tokens among all tokens is between 4% and 7.5% and is quite close for all datasets whereas the part of ending tokens among tokens with full stops varies much more. 

The Gutenberg corpus is chosen as training set as it is possible to extract an important number of sentences and because its rate of ending tokens among tokens with full stops is intermediate between the others.\
The size of the sample train set is taken quite large (30 000 sentences) to provide a sufficient number of occurences of the minority class to avoid overfitting.


The obtained datasets are very imbalanced: the class of tokens not corresponding to a sentence end reprensents > 90% of tokens, which is consistent.
It has therefore been considered to undersample the majority class corresponding to non ending tokens by only considering tokens that contain a full stop (., ?, !) but it degraded the chosen metric on the test set.

### Metric

The considered problem is a binary supervised classification with an imbalanced repartition between the classes. Class 1(True) represents the end sentence tokens.

As a dummy model would have an accuracy >90% in this imbalanced context, hence the relevant metrics are the precision (part of elements predicted True that are correctly True) and the recall (part of real True elements that are detected by the model) for the minority class (class 1).

Therefore, the chosen metric for the model performance evaluation is the **F1 score** for the minority class (class 1) that equivalently takes into account precision and recall (F1 score = 2(precision*recall)/(precision+recall)). One could also choose to favor the recall or the precision score depending on the problem constraints.

The baseline results for the Class 1 on the different sets are:

| Corpus source | Brown corpus | Treebank corpus (WSJ) | Gutenberg corpus |
| --- | --- | --- | --- |
| F1 score | **0.89** | **0.84** | **0.84** |
| Precision | 1 | 0.97 | 1 |
| Recall | 0.8 | 0.73 | 0.62 |


The baseline gives better results in precision than in recall which is consistent as the rule coverage is narrow.

### Feature extraction

The extracted features are computed for each token (current) and its direct next neighbour:

- contains an uppercase
- contains a question or exclamation mark
- contains a period
- contains a quote
- contains a digit
- token's length
- is a stopword (current)

#### Estimator train

Three estimators corresponding to increasing levels of complexity are used for this first approach:
- Logistic Regression using the scikit-learn implementation
- RandomForest Classifier using the scikit-learn implementation
- XGBoost Classifier using the xgboost package implementation

As the volume of data is important to be processed in local, a Halving cross-validation gridsearch (using the scikit-learn implementation) is used for hyperparameter optimization search.
The RandomForest Classifier best estimator gives the higher F1 score in grid search and is therefore selected as the final trained model's estimator.

### Results
 
The results on the test sets are:
| Corpus source | Brown corpus | Treebank corpus (WSJ) |
| --- | --- | --- | 
| F1 score | **0.97** | **0.92** | 
| Precision | 0.94 | 0.96 |
| Recall | 1 | 0.88 | 

The obtained F1 score are higher than the ones of the baseline on both test sets. The precision scores are comparable to the baseline one's however the recall is more satisfaying.

### Perspectives

To increase the scope to deal with more heterogeneous types of texts but also to develop a multi latin language approach, theses perspectives could be considered:

1. Implementing an unsupervised multi-stage classifier approach such as the [Punkt](https://aclanthology.org/J06-4003.pdf) algorithm: Unsupervised Multilingual Sentence Boundary Detection, Tibor Kiss, Jan Strunk, 2006. The approach uses at first likehood ratios to define whether a token is likely to be an abbreviation. Then, for every token with a final period, based on the context, it identifies whether a token is a sentence end.

2. A Neural Network approach such as the [SATZ architecture](https://aclanthology.org/J97-2002.pdf): Adaptive Multilingual Sentence Boundary Disambiguation, David D., Marti A. Hearst. Thanks to POS frequency for the corpus, token vector and context is fed to a fully-connected neural network predicting end of sentence tokens. 

Otherwise, other architecture such as LSTM could be considered.  

## Bonus

Proposition:

- Use of *BeautifulSoup* to extract the document from the html while keeping as much as possible its internal structure.
- From the lists of sentences generated by the model, wrapping with *span* tag around sentences.
- Replacing the previous text by the newly generated paragraph tag.

The code is in the notebook: notebooks/bonus.ipynb (from the root of the project).\
To implement on real-life exemple, we would mainly have to adapt the 'findAll' classe to parse the target text elements. In this example, we consider that the text to tag is only in paragraph tags. 

## Folder structure
```
.
├── notebooks 
|       ├── modeling.ipynb          # notebook with hyperparameter opti and estimator comparison
|       └── bonus.ipynb             # notebook with the proposition for the bonus question
|
├── sbdetection                     # entry point of the package
|       ├── data
|       |     └── models            # trained model serialized as pickle
|       | 
|       ├── models
|       |     ├── MLBased.py        # trained model
|       |     └── RuleBased.py      # rule-based model      
|       |   
|       ├── preprocessing           
|       |     ├── preprocessor.py   # preprocessing
|       |     └── utils.py          # feature extraction utils
|       | 
|       ├── main.py                 # entry point of the CLI
|       └── utils.py                # data import from .txt or nltk corpus 
|
├── tests                           # automated tests (pytest)
├── requirements.txt                # env requirements
└── README.md
```
## Tests

Tests are performed using Pytest. 
The scripts can be found in the ./tests folder.

## Conventions

typing\
pep8
