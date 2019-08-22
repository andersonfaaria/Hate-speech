# NLTK
import nltk
from nltk import word_tokenize
from nltk.stem.porter import *

# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Python, Numpy, Pandas, Regex
import re
import string
import pandas as pd
import numpy as np

# Scitkit-learn
import sklearn
from sklearn.naive_bayes import ComplementNB # Segundo o scitkit learn, foi criado para melhorar o Multinomial NB e é melhor para datasets desbalanceados.
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer # Melhor opção do que fazer CountVectorizer e TfidfTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

################################################
# 1. Definições:
# 1.1 Stemmer
stemmer = PorterStemmer()
# 1.2 Classificador e Vectorizador do nosso pipeline:
pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', ComplementNB()),
    ])
# 1.3 Numero de pastas do k-folds cross validation:
folds = 10
#1.4 Métricas do scoring:
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    }
# 1.5 Parâmetros a serem variados no gridsearch:
params = {
    'classifier__alpha': [.1, .5, 1]
    }    
    
################################################  
# 2. Métodos:
# 2.1 Adaptado dos scripts da aula 16
def negate_sequence(text):
    text2 = ""
    prefix = ""
    for w in re.findall(r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+|[.,;!?]", text):
        if w in ["not", "didn't", "no", "can't"]:
            prefix = "not_"
            continue
        if w in ".,;!?":
            prefix = ""
        text2 += " "+prefix+w        

    return text2

# 2.2 Metodo para remover menções
def remove_mentions(text):
    for w in re.findall("@[\w]*", text):
        text = re.sub(w, '', text)

    return text

# 2.3 Metodo para remover stop words e chamar nosso stemmer (
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(stemmer.stem(token))
    return result
# P.S: Nosso grupo depois descobriu que há funções próprias do NLTK para essas funções de tratamento de dados, e que inclusive poderiam ser incluidas no pipeline.
# Porém, já que tivemos esse trabalho preferimos deixar aqui como aprendizado de implementar a função ao invés de pegá-la pronta.

################################################ 
# 3. Tratando os dados:
X = pd.read_csv('tweets-train.csv') # carrega a base
# 3.1 Remover menções
X['tweet'] = np.vectorize(remove_mentions)(X['tweet'])
# 3.2 Colocar negações (para melhorar análise de sentimento)
X['tweet'] = np.vectorize(negate_sequence)(X['tweet'])
# 3.3 Limpar dados
X['tweet'] = X['tweet'].str.replace("[^a-zA-Z#_]", " ") # filtrando por tudo que não for texto, underscore e hashtag
X['tweet'] = X['tweet'].str.replace("\s+", " ") # filtrando cadeias de espaço e quebras de linha maiores que um.
# 3.4 Remover stop words
#tokens = X['tweet'].apply(lambda w: nltk.word_tokenize(w)) # Tokenizando - não é mais necessário pq a função preprocess ja faz isso com o gensim
#print(tokens.head(10))
# 3.5 Radicalizar (Porter steeming)
tokens = X['tweet'].apply(lambda w: preprocess(w)) # 3.4 e 3.5 Tokeniza, remove os stop words e chama o stemmer.

# Juntando de volta as palavras no tweet:
for w in range(len(tokens)):
    tokens[w] = ' '.join(tokens[w])
X['tweet'] = tokens
y = X['label']
X = X['tweet']
print(X.head(5)) # Mostrando os 5 primeiros tweets (para preview da base final)
################################################ 
# 4. Aplicando o modelo e medindo resultados:
# 4.1 Divide entre treino e teste
skf = StratifiedKFold(folds)
skf.split(X, y) 

# 4.2 Procurando os melhores parametros para nosso modelo
#print(pipe.get_params().keys())
best_attempt = GridSearchCV(pipe, params, cv=skf).fit(X, y)
best_params = best_attempt.best_params_
print(best_params)
pipe.set_params(**best_params)
# 4.3 Validação cruzada
scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False)
# 4.4 Printa resultados
print("    accuracy: %.3f +/- %.3f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
print("    f1: %.3f +/- %.3f" % (scores['test_f1'].mean(), scores['test_f1'].std()))