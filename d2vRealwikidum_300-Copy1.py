
# coding: utf-8

# In[1]:

import logging
import os
import gensim
import smart_open
import  collections
import random
import warnings 
import logging
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')


# In[2]:

#Set up logging configurations  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[3]:

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        with open(self.filename, 'r') as f:
            for uid, line in enumerate(f):
                
                yield LabeledSentence(words=line.split(), tags=['TXT_%s' % uid])

sentences = LabeledLineSentence("/home/fatemeh/Documents/python/dnn-understanding/code/wiki.en.text")

print(sentences)


# In[4]:

model = gensim.models.doc2vec.Doc2Vec(size=300, min_count=5, iter=20)
#(size=400, min_count=3, iter=55)    

model.build_vocab(sentences)

print("vocab done")

model.train(sentences, total_examples=model.corpus_count)

print("train")

model.save("doc2vec_wiki_D_300.txt")


# In[8]:

model.infer_vector(['only', 'you', 'can'])


# In[9]:

print (model.most_similar("fires"))


# In[10]:

print (model["fires"])


# In[11]:

print (model.most_similar('election', topn=10))


# In[13]:

model.most_similar('absurdity', topn=3)


# In[14]:

model.most_similar('manner', topn=4)


# In[16]:

def print_accuracy(model, questions_file):
    print('Evaluating...\n')
    acc = model.accuracy(questions_file)

    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100*float(sem_correct)/sem_total
    print('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))
    
    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc)-1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5,len(acc)-1))
    syn_acc = 100*float(syn_correct)/syn_total
    print('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return (sem_acc, syn_acc)


# In[17]:

word_analogies_file = '/home/fatemeh/Documents/python/dnn-understanding/code/questions-words.txt'
accuracies = []


# In[ ]:

print('Accuracy for word2vec:')
accuracies.append(print_accuracy(model,word_analogies_file))


# In[ ]:

model.evaluate_word_pairs('/home/fatemeh/Documents/python/dnn-understanding/code/accuracy/eval/eval.tab',case_insensitive=True,delimiter='\t', restrict_vocab=30000, dummy4unknown=False)


# In[ ]:



