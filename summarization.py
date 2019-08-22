from sklearn.externals import joblib

import nltk
nltk.download('punkt')
import re
from xml.etree import ElementTree
from nltk import sent_tokenize,word_tokenize
import os
import pandas as pd
import numpy as np
import json
from numpy import dot
from numpy.linalg import norm

def create_test_set():
    '''
    create the test set
    :return: the DataFrame that contains the test set
    '''
    xml_files = os.listdir('test_papers')
    list_summaries = list()
    dict_of_sentences = dict()
    for f in xml_files:
        file = ElementTree.parse("test_papers/"+f)
        #file = ElementTree.parse("xml/" + f)
        root = file.getroot()
        #print(f)
        sections = dict()

        for div in file.getiterator(tag="{http://www.tei-c.org/ns/1.0}div"):

            sec = ""
            for head in div.iter('{http://www.tei-c.org/ns/1.0}head'):
                #print(head.text)
                if head.text not in sections:
                    sections[head.text] = ""
                    sec = head.text

            text = ""
            for p in div.iter('{http://www.tei-c.org/ns/1.0}p'):
                #print(p.text)
                text += p.text
            sections[sec] = text
        list_summaries.append(sections)
        dict_of_sentences[f] = sections

    dict_of_all_the_sentences = {
        'sentence':[],
        'previous':[],
        'next':[],
        'section':[],
        'paper':[],
        'length':[]
    }



    sentences_list = ["first sentence", "last sentence"]

    for pdf, section in dict_of_sentences.items():
        print(pdf)
        for head, text in section.items():
            sentences = sent_tokenize(text)

            for index, sentence in enumerate(sentences):

                if sentence in dict_of_all_the_sentences['sentence']:
                    continue

                dict_of_all_the_sentences['paper'].append(pdf)
                dict_of_all_the_sentences['length'].append(len(word_tokenize(sentence)))
                sect = re.sub(r'[^A-Za-z0-9]+', " ", head).lstrip().lower()
                dict_of_all_the_sentences['section'].append(sect)
                if sect not in sentences_list:
                    sentences_list.append(sect)
                if index == 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    dict_of_all_the_sentences['previous'].append("first sentence")
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)
                elif index > 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    pre_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index - 1)]).lstrip().lower())
                    dict_of_all_the_sentences['previous'].append(pre_sen)
                    if pre_sen not in sentences_list:
                        sentences_list.append(pre_sen)
                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)


    test_set = pd.DataFrame(dict_of_all_the_sentences['sentence'])
    test_set.rename(index=str, columns={0: "sentence"},inplace=True)
    test_set['previous'] = dict_of_all_the_sentences['previous']
    test_set['next'] = dict_of_all_the_sentences['next']
    test_set['section'] = dict_of_all_the_sentences['section']
    test_set['length'] = dict_of_all_the_sentences['length']
    test_set['paper'] = dict_of_all_the_sentences['paper']

    print(test_set)

    return test_set


def extract_features(paper):
    '''

    :param paper:
    :return:
    '''

    xml_files = os.listdir('test_papers')
    list_summaries = list()
    dict_of_sentences = dict()
    for f in xml_files:
        if f.__eq__(paper) is True:
            file = ElementTree.parse("test_papers/" + f)
            root = file.getroot()
            sections = dict()

            for div in file.getiterator(tag="{http://www.tei-c.org/ns/1.0}div"):

                sec = ""
                for head in div.iter('{http://www.tei-c.org/ns/1.0}head'):
                    if head.text not in sections:
                        sections[head.text] = ""
                        sec = head.text

                text = ""
                for p in div.iter('{http://www.tei-c.org/ns/1.0}p'):
                    text += p.text
                sections[sec] = text
            list_summaries.append(sections)
            dict_of_sentences[f] = sections

    dict_of_all_the_sentences = {
        'sentence': [],
        'previous': [],
        'next': [],
        'section': [],
        'paper': [],
        'length': []
    }

    sentences_list = ["first sentence", "last sentence"]

    for pdf, section in dict_of_sentences.items():
        print(pdf)
        for head, text in section.items():
            sentences = sent_tokenize(text)

            for index, sentence in enumerate(sentences):

                if sentence in dict_of_all_the_sentences['sentence']:
                    continue

                dict_of_all_the_sentences['paper'].append(pdf)
                dict_of_all_the_sentences['length'].append(len(word_tokenize(sentence)))
                sect = re.sub(r'[^A-Za-z0-9]+', " ", head).lstrip().lower()
                dict_of_all_the_sentences['section'].append(sect)
                if sect not in sentences_list:
                    sentences_list.append(sect)
                if index == 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    dict_of_all_the_sentences['previous'].append("first sentence")
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)
                elif index > 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    pre_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index - 1)]).lstrip().lower())
                    dict_of_all_the_sentences['previous'].append(pre_sen)
                    if pre_sen not in sentences_list:
                        sentences_list.append(pre_sen)
                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)

    features = pd.DataFrame(dict_of_all_the_sentences['sentence'])
    features.rename(index=str, columns={0: "sentence"}, inplace=True)
    features['previous'] = dict_of_all_the_sentences['previous']
    features['next'] = dict_of_all_the_sentences['next']
    features['section'] = dict_of_all_the_sentences['section']
    features['length'] = dict_of_all_the_sentences['length']
    features['paper'] = dict_of_all_the_sentences['paper']

    print(features)

    return features

'''
def extract_embeddings(paper):
    
    extract the BERT embedding of the sentences and rhe sections of the test set
    :return:
    

    test_set = extract_features(paper)#create_test_set()

    test_sentences_list = ['first sentence','last sentence']
    for row in test_set.iterrows():
        sentence = str(row[1][0])
        section = str(row[1][3])
        if sentence not in test_sentences_list:
            test_sentences_list.append(sentence)
        if section not in test_sentences_list:
            test_sentences_list.append(section)


    with open('sentences.txt','w') as f:
        for line in test_sentences_list:
            f.write("%s\n"%(line))

    os.system('python extract_features.py --input_file=sentences.txt --output_file=output_layer.json --vocab_file=D:/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=D:/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=C:/Users/user/PycharmProjects/bert/my_dataset_output/model.ckpt-3000  --layers=-3  --max_seq_length=128 --batch_size=8')

def get_embeddings():

    sentences = dict()
    with open('sentences.txt','r') as f:
        for index,line in enumerate(f):
            sentences[index] = line.strip()


    embeddings = dict()
    with open('output_layer.json', 'r',encoding='utf-8') as f:
        for line in f:
            embeddings[json.loads(line)['linex_index']] = np.asarray(json.loads(line)['features'])

    sentence_emb = dict()

    for key,value in sentences.items():
        sentence_emb[value] = embeddings[key]


    return sentence_emb

'''

def get_embeddings():

    sentences = dict()
    with open('test_sentences_list.txt','r') as f:
        for index,line in enumerate(f):
            sentences[index] = line.strip()


    embeddings = dict()
    with open('test_output_layer_-3.json', 'r',encoding='utf-8') as f:
        for line in f:
            embeddings[json.loads(line)['linex_index']] = np.asarray(json.loads(line)['features'])

    sentence_emb = dict()

    for key,value in sentences.items():
        sentence_emb[value] = embeddings[key]


    return sentence_emb


def summarize(paper):

    loaded_model = joblib.load('summarizer.pkl')
    test_set = extract_features(paper)
    print(test_set)
    bert_emb = get_embeddings()
    length = list()
    sentence_emb = list()
    previous_emb = list()
    section_emb = list()
    next_list = list()
    section_of_sentences = dict()
    jj_redick = 0
    summary_of_sections = dict()


    for row in test_set.iterrows():
        sentence = row[1][0].strip()
        #sentences.append(sentence)
        previous = row[1][1].strip()
        nexts = row[1][2].strip()
        section = row[1][3].strip()
        section_of_sentences[sentence] = section

        if sentence in bert_emb:
            sentence_emb.append(bert_emb[sentence])
        else:
            sentence_emb.append(np.zeros(768))
            jj_redick += 1

        if previous in bert_emb:
            previous_emb.append(bert_emb[previous])
        else:
            previous_emb.append(np.zeros(768))

        if nexts in bert_emb:
            next_list.append(bert_emb[nexts])
        else:
            next_list.append(np.zeros(768))

        if section in bert_emb:
            section_emb.append(bert_emb[section])
        else:
            section_emb.append(np.zeros(768))

        length.append(row[1][4])



    next_emb = np.asarray(next_list)
    previous_emb = np.asarray(previous_emb)
    section_emb = np.asarray(section_emb)
    length = np.asarray(length)
    features = np.concatenate([sentence_emb, previous_emb, next_emb, section_emb], axis=1)
    features = np.column_stack([features, length])
    predictions = loaded_model.predict_proba(features)

    for index, pred in enumerate(predictions):

        sentence = str(test_set.iloc[index,0]).strip()
        if pred[1] > 0.8:
            section = section_of_sentences[sentence]
            if section not in summary_of_sections:
                summary_of_sections[section] = sentence + ". "
            else:
                summary_of_sections[section] += sentence + ". "

    return summary_of_sections



'''
def summarize(paper):
    loaded_model = joblib.load('summarizer.pkl')

    test_set = extract_features(paper)
    extract_embeddings(paper)
    bert_emb = get_embeddings()

    cosine = list()
    length = list()
    sentence_emb = list()
    previous_emb = list()
    section_emb = list()
    next_list = list()
    sentence_proba = pd.DataFrame()
    nn_sentence_proba = pd.DataFrame()
    jj_redick = 0
    sentences = list()
    section_of_sentences = dict()
    for row in test_set.iterrows():
        sentence = row[1][0].strip()
        sentences.append(sentence)
        previous = row[1][1].strip()
        nexts = row[1][2].strip()
        section = row[1][3].strip()
        section_of_sentences[sentence] = section

        if sentence in bert_emb:
            sentence_emb.append(bert_emb[sentence])
        else:
            sentence_emb.append(np.zeros(768))
            jj_redick += 1

        if previous in bert_emb:
            previous_emb.append(bert_emb[previous])
        else:
            previous_emb.append(np.zeros(768))

        if nexts in bert_emb:
            next_list.append(bert_emb[nexts])
        else:
            next_list.append(np.zeros(768))

        if section in bert_emb:
            section_emb.append(bert_emb[section])
        else:
            section_emb.append(np.zeros(768))

        length.append(row[1][4])

    sentence_proba['sentence'] = sentences
    sentences_set = set()
    nn_sentence_proba['sentence'] = sentences

    summary_text = ""
    sentence_emb = np.asarray(sentence_emb)
    next_emb = np.asarray(next_list)
    previous_emb = np.asarray(previous_emb)
    section_emb = np.asarray(section_emb)
    length = np.asarray(length)
    features = np.concatenate([sentence_emb, previous_emb, next_emb, section_emb], axis=1)
    features = np.column_stack([features, length])

    predictions = loaded_model.predict_proba(features)
    no_sen = 0
    log_preds = list()
    for i in predictions:
        log_preds.append(i[1])

    sentence_proba['probability'] = log_preds
    sentence_proba.sort_values(by=['probability'], inplace=True, ascending=False)

    summary_of_sections = dict()
    for row in sentence_proba.iterrows():
        sentence = row[1][0].strip()
        section = section_of_sentences[sentence]

        if (len(word_tokenize(summary_text)) < 400):
            sent_len = len(word_tokenize(sentence))
            if (len(word_tokenize(summary_text)) + sent_len) > 400:
                continue
            max_cosine = 0.0
            if len(sentences_set) > 0:
                for sen in sentences_set:
                    cos_sim = dot(bert_emb[sen], bert_emb[sentence]) / (norm(bert_emb[sentence]) * norm(bert_emb[sen]))
                    if max_cosine < cos_sim:
                        max_cosine = cos_sim
                cosine.append(max_cosine)
                if max_cosine < 0.90:
                    if section not in summary_of_sections:
                        summary_of_sections[section] = sentence + "."
                    else:
                        summary_of_sections[section] += sentence + "."
                    summary_text += sentence + " "
                    sentences_set.add(sentence)
                    no_sen += 1
            else:
                if section not in summary_of_sections:
                    summary_of_sections[section] = sentence + "."
                else:
                    summary_of_sections[section] += sentence + "."
                summary_text += sentence + " "
                sentences_set.add(sentence)
                no_sen += 1

    print(jj_redick)
    return summary_of_sections
'''

def get_paper_name(paper):
    file = ElementTree.parse("test_papers/" + paper)

    titles = list()
    for item in file.getiterator('{http://www.tei-c.org/ns/1.0}title'):
        titles.append(item)

    return titles[0].text



if __name__ == '__main__':

    print(get_paper_name('400.tei.xml'))
    #a = summarize('400.tei.xml')
    #summaries = third_approach()
    #for paper,summary in summaries.items():
      #  print(paper,summary)
