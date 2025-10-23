import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

# ------------------------
# SENTENCE TOKENIZATION
# ------------------------
from nltk.tokenize import sent_tokenize
sentence='Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence, and linguistics to enable computers to understand, process, and generate human language in a meaningful way. It involves using machine learning techniques to facilitate communication between humans and machines, making it possible for computers to interpret and respond to written and spoken language. NLP is widely used in applications such as chatbots, translation services, and voice recognition systems'

print(sent_tokenize(sentence))
"""
the output is :-

['Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence, and linguistics to enable computers to understand, process, and generate human language in a meaningful way.', 'It involves using machine learning techniques to facilitate communication between humans and machines, making it possible for computers to interpret and respond to written and spoken language.', 'NLP is widely used in applications such as chatbots, translation services, and voice recognition systems']
"""

# ------------------------
# WORD TOKENIZATION
# ------------------------
from nltk.tokenize import word_tokenize

tokens=word_tokenize(sentence)
print(tokens)
"""
['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'is', 'a', 'field', 'that', 'combines', 'computer', 'science', ',', 'artificial', 'intelligence', ',', 'and', 'linguistics', 'to', 'enable', 'computers', 'to', 'understand', ',', 'process', ',', 'and', 'generate', 'human', 'language', 'in', 'a', 'meaningful', 'way', '.', 'It', 'involves', 'using', 'machine', 'learning', 'techniques', 'to', 'facilitate', 'communication', 'between', 'humans', 'and', 'machines', ',', 'making', 'it', 'possible', 'for', 'computers', 'to', 'interpret', 'and', 'respond', 'to', 'written', 'and', 'spoken', 'language', '.', 'NLP', 'is', 'widely', 'used', 'in', 'applications', 'such', 'as', 'chatbots', ',', 'translation', 'services', ',', 'and', 'voice', 'recognition', 'systems']
"""

# -------------------------------------------------------
# Normalize the tokens - maybe convert to lower case
# -------------------------------------------------------


lower_tokens=[t.lower() for t in tokens] # convert all tokens to lowercase
print(lower_tokens)
"""
['natural', 'language', 'processing', '(', 'nlp', ')', 'is', 'a', 'field', 'that', 'combines', 'computer', 'science', ',', 'artificial', 'intelligence', ',', 'and', 'linguistics', 'to', 'enable', 'computers', 'to', 'understand', ',', 'process', ',', 'and', 'generate', 'human', 'language', 'in', 'a', 'meaningful', 'way', '.', 'it', 'involves', 'using', 'machine', 'learning', 'techniques', 'to', 'facilitate', 'communication', 'between', 'humans', 'and', 'machines', ',', 'making', 'it', 'possible', 'for', 'computers', 'to', 'interpret', 'and', 'respond', 'to', 'written', 'and', 'spoken', 'language', '.', 'nlp', 'is', 'widely', 'used', 'in', 'applications', 'such', 'as', 'chatbots', ',', 'translation', 'services', ',', 'and', 'voice', 'recognition', 'systems']
"""
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
filtered_tokens=[t for t in lower_tokens if t.isalpha() and t not in stop_words]
print(filtered_tokens)

from nltk.stem import PorterStemmer