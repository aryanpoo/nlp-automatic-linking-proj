from collections import defaultdict
from os import path
import json
import mlflow
import numpy as np

from tqdm import tqdm
import spacy
import pdfplumber
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

from downloads import get_fn_of_doi, BASE_DATA_DIR, DATASET_FN, DATA_PAPERS_DIR, RESEARCH_PAPERS_DIR

def get_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        print(f'Model {model_name} not found. Downloading...')
        spacy.cli.download(model_name)
        print(f'Model {model_name} downloaded.')

    return spacy.load(model_name)


def save_entities_to_file(entities, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=4)


def load_entities_from_file(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_pdf(fn):
    text = ""
    with pdfplumber.open(fn) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_text_html(fn):
    with open(fn, "r", encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def extract_text_txt(fn):
    with open(fn, "r", encoding='utf-8') as f:
        return f.read()
    

def extract_and_save_entities(df, paper_type, nlp, remove_nums, text_type):
    """Extract and save named entities.
    
    Args:
        df: Linking dataset as a DataFrame.
        paper_type: Type of papers ('data' or 'research') which their entities should be extracted.
        nlp: A spaCy model.
        remove_nums: A Boolean flag indicating whether numerical entities should be removed from the resulting list.
        text_type: Either 'full-text' for article's full text or 'abstract' for article abstracts.

    """
    assert paper_type in ['data', 'research']
    assert text_type in ['full-text', 'abstract']
    entities_fn = path.join(BASE_DATA_DIR, f'{paper_type}_papers_entities_remove_nums={remove_nums}_text_type={text_type}.json')
    if path.exists(entities_fn):
        return load_entities_from_file(entities_fn)

    entities = dict()
    print(f'Extracting entities of {paper_type} papers ...')
    for _, row in tqdm(df.iterrows()):
        doi = row[f'{paper_type}_paper_doi']
        fn = get_fn_of_doi(doi, DATA_PAPERS_DIR if paper_type == 'data' else RESEARCH_PAPERS_DIR, already_exists=True)
        
        rest, ext = fn.rsplit('.', 1)
        if text_type == 'abstract':
            ext = 'txt'
            fn = rest + '.abstract.txt'

        if ext == 'pdf':
            text = extract_text_pdf(fn)
        elif ext == 'html':
            text = extract_text_html(fn)
        elif ext == 'txt':
            text = extract_text_txt(fn)
        else:
            print(f'Unsupported file format: {ext}')
            continue

        entities[doi] = extract_entities(nlp, text, remove_nums)
    save_entities_to_file(entities, entities_fn)
    return entities

def extract_entities(nlp, text, remove_nums):
    """Extract named entities from a piece of text.
    
    Args:
        nlp: A spaCy model.
        text: The input text from which named entities will be extracted.
        remove_nums: A Boolean flag indicating whether numerical entities should be removed from the resulting list.
    """
    if not remove_nums:
        return [e.text for e in nlp(text).ents]
    else:
        return [e.text for e in nlp(text).ents if not e.text.isdigit()]


def link_a_paper_by_ner(src_doi, src_type, entities_dict):
    """Rank corresponding papers of a given paper by NER method.
    
    Args:
        src_doi: DOI of the paper that its corresponding papers are found.
        src_type: Type of the paper ('data' or 'research') that its corresponding papers should be found. 
        entities_dict: A dictionary with keys=('data' or 'research'). Each value is a dictionary as returned by extract_and_save_entities.
    """
    assert src_type in ['data', 'research']
    other_type =  'data' if src_type == 'research' else 'research'
    ranks = dict()
    src_entities = entities_dict[src_type][src_doi]
    for other_doi, other_entities in entities_dict[other_type].items():
        common_entities = set(src_entities) & set(other_entities)
        ranks[other_doi] = len(common_entities)
    return dict(sorted(ranks.items(), key=lambda item: item[1], reverse=True))


def ner_method_scores(src_type, src_dois, entities_dict, df, mlflow_on, remove_nums, text_type):
    """Apply NER method to the whole dataset. 
    
    Args:
        df: Linking dataset as a DataFrame.
        src_type: Type of source papers ('data' or 'research') that their corresponding papers are found.
    """
    assert src_type in ['data', 'research']
    other_type =  'data' if src_type == 'research' else 'research'
    correct_ranks = []
    for src_doi in src_dois:
        rank = link_a_paper_by_ner(src_doi, src_type, entities_dict)
        rank_dois = list(rank.keys())
        ans_doi = df[df[f'{src_type}_paper_doi']==src_doi].iloc[0][f'{other_type}_paper_doi']
        correct_ranks.append(rank_dois.index(ans_doi))
    
    correct_ranks = np.array(correct_ranks) + 1
    mrr = (1/correct_ranks).mean()
    rr_std = (1/correct_ranks).std()
    print(f'Finding corresponding papers of {src_type} papers')
    print(f'Reciprocal Rank: MMR={mrr:.5f}, STD={rr_std:.2f}')
    fig = plt.figure()
    ax = fig.gca()
    labels, counts = np.unique(correct_ranks, return_counts=True)
    ax.bar(labels, counts, align='center')
    ax.set_xticks(labels)
    ax.set_title(f'Histogram of correct-answer-rank for\nfinding corresponding papers of {src_type} papers (NER)')
    
    if mlflow_on:
        with mlflow.start_run():
            mlflow.log_param('method', 'NER')
            mlflow.log_param('remove_nums', remove_nums)
            mlflow.log_param('text_type', text_type)
            mlflow.log_param('src_type', src_type)
            mlflow.log_metric('mmr', mrr)
            mlflow.log_metric('rr_std', rr_std)
            mlflow.log_figure(fig, 'hist-of-ranks.png')

    return mrr, rr_std

####################################################
MLFLOW_ON = not True
SPACY_MODEL_NAME = 'en_core_web_sm'
CV_NUM_FOLDS = 5
nlp = get_spacy_model(SPACY_MODEL_NAME)

df = pd.read_csv(DATASET_FN, index_col=0)
print('Total number of pairs of data/research papers in the dataset:', len(df))

df = df[(~ df['data_paper_fn'].isna()) & (~ df['research_paper_fn'].isna())]
print('Number of pairs with both files present:', len(df))

"""
Note on the cross-validation like method employed:
text_type and remove_nums are hyperparameters. As they have no parameters, the algorithms
do not necessitate a conventional "training phase" but requires an optimal selection of these hyperparameters.
Given the limited size of the dataset, a method analogous to cross-validation is employed. For
each possible combination of hyperparameters, the algorithm is tested on K-1 folds of the data,
This procedure, though reminiscent of cross-validation, essentially serves as a
robust performance evaluation mechanism for each hyperparameter configuration.
The set of hyperparameters that yield the highest average performance across all folds is then selected as the optimal choice."""
results = defaultdict(dict)
for text_type in ['full-text', 'abstract']:
    for remove_nums in [False, True]:
        print(f'\n\n####### {text_type} {remove_nums} ##########')
        entities_dict = dict()
        for paper_type in ['research', 'data']:
            entities_dict[paper_type] = extract_and_save_entities(df, paper_type, nlp, remove_nums, text_type)
        if MLFLOW_ON:
            mlflow.create_experiment('ner-all')
            mlflow.set_experiment('raw-ner')
        for src_type in ['data', 'research']:
            all_src_dois = list(entities_dict[src_type])
            CV_LEN = len(all_src_dois) // CV_NUM_FOLDS
            mrrs, rr_stds = [], []
            for k in range(CV_NUM_FOLDS):
                if k == CV_NUM_FOLDS - 1:
                    src_dois = all_src_dois[k*CV_LEN:]
                else:
                    src_dois = all_src_dois[k*CV_LEN:(k+1)*CV_LEN]
                mrr, rr_std = ner_method_scores(src_type, src_dois, entities_dict, df, MLFLOW_ON, remove_nums, text_type)
                mrrs.append(mrr)
                rr_stds.append(rr_std)
            results[(text_type, remove_nums)][src_type] = (np.array(mrrs).mean(), np.array(rr_stds).mean())

print(results)
# plt.show()