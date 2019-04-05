#!/usr/bin/env python
# coding: utf-8

# In[7]:


from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingStanford
from swisscom_ai.research_keyphrase.util.fileIO import read_file
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


# load pos tagger
def load_local_pos_tagger(lang):
    assert (lang in ['en', 'de', 'fr']), "Only english 'en', german 'de' and french 'fr' are handled"
    #jar_path = config_parser.get('STANFORDTAGGER', 'jar_path')
    stanford_ner_dir = './stanford-postagger-full-2018-10-16/'
    model_directory_path= './stanford-postagger-full-2018-10-16/models/'
    jar_path= stanford_ner_dir + 'stanford-postagger.jar'

    return PosTaggingStanford(jar_path, model_directory_path, lang=lang)


# In[11]:


# Maximal Marginal Relevance Similarity Calculation
def _MMR(candidates, X, doc_embedd, beta, N):
    """
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates
    :param embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param candidates: list of candidates (string)
    :param X: numpy array with the embedding of each candidate in each row
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of candidates to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """

    N = min(N, len(candidates))
    doc_sim = cosine_similarity(X, doc_embedd)

    doc_sim_norm = doc_sim/np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)

    sim_between = cosine_similarity(X)
    np.fill_diagonal(sim_between, np.NaN)

    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
    sim_between_norm =         0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)

    selected_candidates = []
    unselected_candidates = [c for c in range(len(candidates))]

    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)

    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)

        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1:
            dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)
    return selected_candidates
    


# In[19]:


import sent2vec

#Load sent2vec model
model = sent2vec.Sent2vecModel()
model.load_model('./sent2vec/wiki_unigrams.bin')


# In[21]:


import argparse
from configparser import ConfigParser
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates, extract_sent_candidates
pos_tagger = load_local_pos_tagger('en')
count = 0
tp = 0
r = 0
p = 0
F = 0
for file in os.listdir("./Dataset/Hulth2003/Test/"):
    # select only 100 sample text to test
    if(count == 2): break
    if file.endswith(".abstr"):
        path = os.path.join("./Dataset/Hulth2003/Test/", file)
        print('Processing: ',file)
        f=open(path,'r').read()
        f = f.replace("\n", " ")
        f = f.replace("\t", " ")
        raw_text = f
        #print(raw_text)
        tagged = pos_tagger.pos_tag_raw_text(raw_text)
        text_obj = InputTextObj(tagged, 'en')
        # List of candidates based on PosTag rules
        candidates = np.array(extract_candidates(text_obj))  
        # Remove Duplicates
        candidates = list(set(candidates))
        #print(candidates)
        tagged = text_obj.filtered_pos_tagged
        tokenized_doc_text = ' '.join(token[0].lower() for sent in tagged for token in sent)
        #print(tokenized_doc_text)
        document = tokenized_doc_text
        #print(document)
        doc = model.embed_sentence(document) 
        #print(doc)
        #print(doc.shape)
        doc = doc.reshape(1, -1)
        #print(doc.shape)
        score = []
        for x in candidates:
            sentence = x
            res = model.embed_sentence(x) 
            score.append(res)
        #print(score)
        #print(score.shape)
        score = np.asarray(score)
        # result contains index of the top 15 candidate keyphrases
        result = _MMR(candidates, score, doc, 1, 15)
        #print(result)
        pred_kp = []
        for i in range(len(result)):
            pred_kp.append(candidates[result[i]].lower())
        print('predicted keyphrase: ',pred_kp)
        f2=open("./Dataset/Hulth2003/Test/" + file[:-6]+".uncontr",'r').read()
        f2 = f2.replace("\n", "")
        f2 = f2.replace("\t", " ")
        f2 = f2.split('; ')
        actual_kp = f2
        for x in actual_kp:
            x_lower = x.lower()
            if raw_text.find(x_lower) == -1:
                actual_kp.remove(x)
        for x in actual_kp:
            x = x.lower()
        actual_kp = list(set(actual_kp))
        print(actual_kp)
        intersect = list(set(pred_kp).intersection(actual_kp))
        tp = tp + len(intersect)
        #print(intersect)
        #print(tp)
        r = r + len(pred_kp)
        #print(r)
        p = p + len(actual_kp)
        #print(p)
        count = count + 1
        
precision = tp / p
recall = tp / r
F = 2*precision*recall / (precision + recall)
print(precision)
print(recall)
print(F)



# In[ ]:




