import nltk
import spacy
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download('en_core_web_sm')

# ------------------------
# SENTENCE TOKENIZATION
# ------------------------
from nltk.tokenize import sent_tokenize
sentence='Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence, and linguistics to enable computers to understand, process, and generate human language in a meaningful way. It involves using machine learning techniques to facilitate communication between humans and machines, making it possible for computers to interpret and respond to written and spoken language. NLP is widely used in applications such as chatbots, translation services, and voice recognition systems'

# print(sent_tokenize(sentence))
"""
the output is :-

['Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence, and linguistics to enable computers to understand, process, and generate human language in a meaningful way.', 'It involves using machine learning techniques to facilitate communication between humans and machines, making it possible for computers to interpret and respond to written and spoken language.', 'NLP is widely used in applications such as chatbots, translation services, and voice recognition systems']
"""

# ------------------------
# WORD TOKENIZATION
# ------------------------
from nltk.tokenize import word_tokenize

tokens=word_tokenize(sentence)
# this means we do niot want to tokenize on the basis of the senetnces , we want to do on the basis of the words 
# print(tokens)
"""
['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'is', 'a', 'field', 'that', 'combines', 'computer', 'science', ',', 'artificial', 'intelligence', ',', 'and', 'linguistics', 'to', 'enable', 'computers', 'to', 'understand', ',', 'process', ',', 'and', 'generate', 'human', 'language', 'in', 'a', 'meaningful', 'way', '.', 'It', 'involves', 'using', 'machine', 'learning', 'techniques', 'to', 'facilitate', 'communication', 'between', 'humans', 'and', 'machines', ',', 'making', 'it', 'possible', 'for', 'computers', 'to', 'interpret', 'and', 'respond', 'to', 'written', 'and', 'spoken', 'language', '.', 'NLP', 'is', 'widely', 'used', 'in', 'applications', 'such', 'as', 'chatbots', ',', 'translation', 'services', ',', 'and', 'voice', 'recognition', 'systems']
"""

# -------------------------------------------------------
# Normalize the tokens - maybe convert to lower case
# -------------------------------------------------------


lower_tokens=[t.lower() for t in tokens] # convert all tokens to lowercase
# print(lower_tokens)
"""
['natural', 'language', 'processing', '(', 'nlp', ')', 'is', 'a', 'field', 'that', 'combines', 'computer', 'science', ',', 'artificial', 'intelligence', ',', 'and', 'linguistics', 'to', 'enable', 'computers', 'to', 'understand', ',', 'process', ',', 'and', 'generate', 'human', 'language', 'in', 'a', 'meaningful', 'way', '.', 'it', 'involves', 'using', 'machine', 'learning', 'techniques', 'to', 'facilitate', 'communication', 'between', 'humans', 'and', 'machines', ',', 'making', 'it', 'possible', 'for', 'computers', 'to', 'interpret', 'and', 'respond', 'to', 'written', 'and', 'spoken', 'language', '.', 'nlp', 'is', 'widely', 'used', 'in', 'applications', 'such', 'as', 'chatbots', ',', 'translation', 'services', ',', 'and', 'voice', 'recognition', 'systems']
"""
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
# print(stop_words)
"""
{'there', "won't", 'were', "it'd", 'then', "shan't", 'some', 'no', 'further', 'from', "don't", 'at', 're', "she's", "we've", 'didn', 'ours', "we'd", 'aren', "they're", 'against', "we'll", 'all', "we're", "he'd", "mightn't", 'ain', "that'll", "you'd", 'had', 'it', 'most', "it'll", 'the', 'they', 'yours', 'yourself', 'once', "should've", 'that', "they'll", 'is', 'not', 'weren', 'mightn', 'this', 'above', 'how', 'isn', 'y', 'off', "shouldn't", 'about', 'shan', 'nor', 'be', 'until', 'those', "he's", 'only', 'over', 'same', 'me', "she'd", "you're", 'was', 'if', "aren't", 'd', 'yourselves', 'doesn', 'my', "she'll", 'haven', 'its', 'here', 'i', 'just', 'other', 'but', 'into', 'he', 'myself', 'again', 'a', 'hers', 'been', 'wasn', 's', 'them', "needn't", 'wouldn', 'she', 'who', 'after', "they'd", 'what', "it's", 'between', 'do', 'when', "hadn't", 'doing', 'so', 't', 'does', 'any', 'her', 'on', 'himself', "doesn't", 'have', 'before', 'with', 'for', 'needn', 'too', 'ma', 'very', 'having', 'should', 'in', 'did', 'out', 'themselves', 'through', "he'll", 'because', "didn't", 'to', 'our', "wasn't", 'theirs', 'down', 'will', "i'll", "isn't", 'by', 'hadn', 'during', 'itself', 'whom', 'o', 'you', 'herself', 'ourselves', 'while', 'these', 'and', 'hasn', 'as', 'below', 'his', "i've", 'shouldn', 'such', "couldn't", "weren't", "wouldn't", "you've", "mustn't", 'your', 'more', 'now', "they've", "you'll", 'm', 'up', 'couldn', 'am', 'don', 'under', 'of', 'their', 'being', 'has', "i'm", 'both', 'or', "haven't", 'him', 'each', 'few', "hasn't", 'll', 'where', 'can', 'are', "i'd", 'own', 'than', 've', 'why', 'which', 'an', 'mustn', 'we', 'won'}
"""
# now we already know taht we do not need stopwords as they don't carry any semantic meaning 

# so we filter those words 

filtered_tokens=[t for t in lower_tokens if t.isalpha() and t not in stop_words]
# print('filtered tokens=\n',filtered_tokens)
"""
 ['natural', 'language', 'processing', 'nlp', 'field', 'combines', 'computer', 'science', 'artificial', 'intelligence', 'linguistics', 'enable', 'computers', 'understand', 'process', 'generate', 'human', 'language', 'meaningful', 'way', 'involves', 'using', 'machine', 'learning', 'techniques', 'facilitate', 'communication', 'humans', 'machines', 'making', 'possible', 'computers', 'interpret', 'respond', 'written', 'spoken', 'language', 'nlp', 'widely', 'used', 'applications', 'chatbots', 'translation', 'services', 'voice', 'recognition', 'systems']
"""

from nltk.stem import PorterStemmer,WordNetLemmatizer

stemmer=PorterStemmer()
stemmed_tokens=[stemmer.stem(t) for t in filtered_tokens]
# print('stemmed tokens=\n',stemmed_tokens)
"""
 ['natur', 'languag', 'process', 'nlp', 'field', 'combin', 'comput', 'scienc', 'artifici', 'intellig', 'linguist', 'enabl', 'comput', 'understand', 'process', 'gener', 'human', 'languag', 'meaning', 'way', 'involv', 'use', 'machin', 'learn', 'techniqu', 'facilit', 'commun', 'human', 'machin', 'make', 'possibl', 'comput', 'interpret', 'respond', 'written', 'spoken', 'languag', 'nlp', 'wide', 'use', 'applic', 'chatbot', 'translat', 'servic', 'voic', 'recognit', 'system']
"""
lemmatizer=WordNetLemmatizer()
lemmatized_tokens=[lemmatizer.lemmatize(t) for t in filtered_tokens]
# print('lemmatized tokens=\n',lemmatized_tokens)

"""
 ['natural', 'language', 'processing', 'nlp', 'field', 'combine', 'computer', 'science', 'artificial', 'intelligence', 'linguistics', 'enable', 'computer', 'understand', 'process', 'generate', 'human', 'language', 'meaningful', 'way', 'involves', 'using', 'machine', 'learning', 'technique', 'facilitate', 'communication', 'human', 'machine', 'making', 'possible', 'computer', 'interpret', 'respond', 'written', 'spoken', 'language', 'nlp', 'widely', 'used', 'application', 'chatbots', 'translation', 'service', 'voice', 'recognition', 'system']
"""



# -----------------------------
# -----------------------------
# Implemantation using Spacy
# -----------------------------
# -----------------------------

nlp=spacy.load('en_core_web_sm')
doc=nlp(sentence)
# print(doc)
'''
Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence, and linguistics to enable computers to understand, process, and generate human language in a meaningful way. It involves using machine learning techniques to facilitate communication between humans and machines, making it possible for computers to interpret and respond to written and spoken language. NLP is widely used in applications such as chatbots, translation services, and voice recognition systems
'''
# the sentence is converted in a doc . This is a Object Oriented approach . This doc is an iterable . We can iterate over it and print each word(This is same as word tokenisation or sentence tokenization but an easier method using this library)
words=[]
for sentence in doc.sents:
    for word in sentence:
        # print(word)
        words.append(word)


# the ouitpout is 
'''
Natural
Language
Processing
(
NLP
)
is
a
field
that
combines
computer
science
,
artificial
intelligence
,
and
linguistics
to
enable
computers
to
understand
,
process
,
and
generate
human
language
in
a
meaningful
way
.
It
involves
using
machine
learning
techniques
to
facilitate
communication
between
humans
and
machines
,
making
it
possible
for
computers
to
interpret
and
respond
to
written
and
spoken
language
.
NLP
is
widely
used
in
applications
such
as
chatbots
,
translation
services
,
and
voice
recognition
systems
'''



# ------------------------------
# ------------------------------
#       WORD EMBEDDINGS 
# ------------------------------
# ------------------------------
from gensim.models import Word2Vec
sentences=[ ['natural', 'language', 'processing', 'nlp', 'field', 'combines', 'computer', 'science', 'artificial', 'intelligence', 'linguistics', 'enable', 'computers', 'understand', 'process', 'generate', 'human', 'language', 'meaningful', 'way', 'involves', 'using', 'machine', 'learning', 'techniques', 'facilitate', 'communication', 'humans', 'machines', 'making', 'possible', 'computers', 'interpret', 'respond', 'written', 'spoken', 'language', 'nlp', 'widely', 'used', 'applications', 'chatbots', 'translation', 'services', 'voice', 'recognition', 'systems']]

model=Word2Vec(sentences,vector_size=50,window=3,min_count=1,workers=4)
for token in sentences:
    print(model.wv[token] )
'''
[[ 0.00266505  0.01308171  0.01996921 ... -0.00789252 -0.0188609
  -0.00154995]
 [-0.001071    0.0004435   0.01020316 ...  0.01920662  0.00997574
   0.01848397]
 [ 0.01217073 -0.01355699  0.00137812 ... -0.0033387  -0.01890357
  -0.00520794]
 ...
 [ 0.00287573 -0.00531005 -0.01415037 ...  0.00100666  0.01643031
  -0.01403381]
 [ 0.00018833  0.00611603 -0.01362436 ... -0.00542776 -0.00873085
  -0.00204642]
 [ 0.01563174 -0.01903477 -0.00041815 ... -0.0048018  -0.01901591
   0.00901127]]
'''