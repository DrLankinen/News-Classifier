#!/usr/bin/env python
# coding: utf-8

# ## Data

# In[1]:


import os
import glob
import io
import pandas as pd
from sklearn.utils import shuffle


# In[2]:


path = "/home/elias/Documents/Data/news_categories/bbc-fulltext/bbc/train"
examples = []
for i,label in enumerate(['business','entertainment','politics','sport','tech']):
    for fname in glob.iglob(os.path.join(path, label, '*.txt')):
        try:
            with io.open(fname, 'r', encoding="utf-8") as f:
                text = f.read()
            examples.append([i,text])
        except:
            print("error reading file ", fname)
            continue
path = "/home/elias/Documents/Data/news_categories/bbc-fulltext/bbc/test"
for i,label in enumerate(['business','entertainment','politics','sport','tech']):
    for fname in glob.iglob(os.path.join(path, label, '*.txt')):
        try:
            with io.open(fname, 'r', encoding="utf-8") as f:
                text = f.read()
            examples.append([i,text])
        except:
            print("error reading file ", fname)
            continue


# In[3]:


df = shuffle(pd.DataFrame(examples,columns=['label','text']))


# In[4]:


df = df.reset_index(drop=True)


# In[5]:


df.head(3)


# In[6]:


df.to_pickle('bbc_data_df')


# ## Imports

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from newspaper import Article
from fastai import *
from fastai.text import *


# - 0 -> Business
# - 1 -> Entertainment
# - 2 -> Politics
# - 3 -> Sport
# - 4 -> Tech

# In[2]:


import fastai
fastai.__version__


# ## Data processing

# In[3]:


df = pd.read_pickle('bbc_data_df')


# In[4]:


df = shuffle(df).reset_index(drop=True)


# In[5]:


df.head(3)


# In[6]:


train_pro = 0.9

train_df = df[:round(len(df)*train_pro)]
train_df = train_df.reset_index(drop=True)
valid_df = df[round(len(df)*train_pro):]
valid_df = valid_df.reset_index(drop=True)


# In[7]:


len(train_df),len(valid_df)


# In[8]:


train_df.head(3)


# In[9]:


valid_df.head(3)


# In[10]:


train_df['label'].value_counts()


# In[11]:


valid_df['label'].value_counts()


# In[12]:


data_lm = TextLMDataBunch.from_df('./', train_df=train_df, valid_df=valid_df)


# In[13]:


data_lm.show_batch()


# In[14]:


#data_lm.save('tmp_lm')


# In[15]:


#data_lm = TextClasDataBunch.load('./','tmp_lm')


# ## Language model

# In[14]:


learner = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.5)
learner.lr_find()
learner.recorder.plot()


# In[15]:


learner.fit_one_cycle(1,1e-2)
learner.unfreeze()


# In[16]:


learner.fit_one_cycle(1,1e-3)
learner.save_encoder('fine_enc')


# ## Classifier

# In[17]:


data_clas = TextClasDataBunch.from_df('./',train_df=train_df,valid_df=valid_df,vocab=data_lm.train_ds.vocab,bs=32)


# In[20]:


data_clas.export("./export.pkl")


# In[21]:


classifier = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
classifier.load_encoder('fine_enc')


# In[22]:


classifier.lr_find()
classifier.recorder.plot()


# In[23]:


classifier.fit_one_cycle(2,2e-2,moms=(0.8,0.7))
classifier.recorder.plot_losses()


# In[24]:


classifier.save('./model')


# ## Test

# In[3]:


data_clas = TextClasDataBunch.load_empty('./')


# In[4]:


classifier = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
_ = classifier.load('./model')


# In[114]:


url = "https://www.bbc.com/sport/cricket/48582537"


# In[121]:


article = Article(url)
article.download()
article.parse()
article_text = article.text


# In[116]:


article_text = "Yuvraj Singh: India all-rounder retires from international cricket"


# In[117]:


label, number, values = classifier.predict(article_text,own=False)


# In[118]:


values.numpy()


# In[119]:


label


# In[120]:


labels = ['business','entertainment','politics','sport','tech']
for i,v in enumerate(values.numpy()):
    print(labels[i],"-",v)


# In[ ]:





# In[ ]:




