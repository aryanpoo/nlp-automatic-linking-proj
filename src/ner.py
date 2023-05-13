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
    

def extract_and_save_entities(df, paper_type, nlp):
    """Extract and save named entities.
    
    Args:
        df: linking dataset as a DataFrame
        paper_type: type of papers ('data' or 'research') which their entities are extracted
        nlp: an spacy model
    """
    assert paper_type in ['data', 'research']
    entities_fn = path.join(BASE_DATA_DIR, f'{paper_type}_papers_entities.json')
    if path.exists(entities_fn):
        return load_entities_from_file(entities_fn)

    entities = dict()
    print(f'Extrating entities of {paper_type} papers ...')
    for _, row in tqdm(df.iterrows()):
        doi = row[f'{paper_type}_paper_doi']
        fn = get_fn_of_doi(doi, DATA_PAPERS_DIR if paper_type == 'data' else RESEARCH_PAPERS_DIR, already_exists=True)
        file_extension = fn.split('.')[-1]

        if file_extension == 'pdf':
            text = extract_text_pdf(fn)
        elif file_extension == 'html':
            text = extract_text_html(fn)
        elif file_extension == 'txt':
            text = extract_text_txt(fn)
        else:
            print(f'Unsupported file format: {file_extension}')
            continue

        entities[doi] = [e.text for e in nlp(text).ents]
    save_entities_to_file(entities, entities_fn)
    return entities


def link_a_paper_by_ner(src_doi, src_type, entities_dict):
    """Rank corresponding papers of a given paper by NER method.
    
    Args:
        src_doi: doi of the paper that its corresponding papers are found
        src_type: type of the paper ('data' or 'research') that its corresponding papers are found 
        entities_dict: a dictionary with keys=('data' or 'research'). Each value is a dictionary as returned by extract_and_save_entities
    """
    assert src_type in ['data', 'research']
    other_type =  'data' if src_type == 'research' else 'research'
    ranks = dict()
    src_entities = entities_dict[src_type][src_doi]
    for other_doi, other_entities in entities_dict[other_type].items():
        common_entities = set(src_entities) & set(other_entities)
        ranks[other_doi] = len(common_entities)
    return dict(sorted(ranks.items(), key=lambda item: item[1], reverse=True))


def ner_method_scores(src_type, entities_dict, df):
    """Apply NER method to the whole dataset. 
    
    Args:
        df: linking dataset as a DataFrame
        src_type: type of source papers ('data' or 'research') that their corresponding papers are found
    """
    assert src_type in ['data', 'research']
    other_type =  'data' if src_type == 'research' else 'research'
    correct_ranks = []
    for src_doi in entities_dict[src_type]:
        rank = link_a_paper_by_ner(src_doi, src_type, entities_dict)
        rank_dois = list(rank.keys())
        ans_doi = df[df[f'{src_type}_paper_doi']==src_doi].iloc[0][f'{other_type}_paper_doi']
        correct_ranks.append(rank_dois.index(ans_doi))
    
    correct_ranks = np.array(correct_ranks) + 1
    mrr = (1/correct_ranks).mean()
    rr_std = (1/correct_ranks).std()
    print('Finding corresponding papers of {src_type} papers')
    print(f'Reciprocal Rank: MMR={mrr:.2f}, STD={rr_std:.2f}')
    fig = plt.figure()
    ax = fig.gca()
    labels, counts = np.unique(correct_ranks, return_counts=True)
    ax.bar(labels, counts, align='center')
    ax.set_xticks(labels)
    ax.set_title(f'Histogram of correct-answer-rank for\nfinding corresponding papers of {src_type} papers (NER)')
    
    with mlflow.start_run():
        mlflow.log_param('method', 'NER (raw)')
        mlflow.log_param('src_type', src_type)
        mlflow.log_metric('mmr', mrr)
        mlflow.log_metric('rr_std', rr_std)
        mlflow.log_figure(fig, 'hist-of-ranks.png')

    
####################################################
# Download the spaCy model if not already installed
SPACY_MODEL_NAME = 'en_core_web_sm'
nlp = get_spacy_model(SPACY_MODEL_NAME)

df = pd.read_csv(DATASET_FN, index_col=0)
print('Total number of pairs of data/research papers in the dataset:', len(df))

df = df[(~ df['data_paper_fn'].isna()) & (~ df['research_paper_fn'].isna())]
print('Number of pairs with both files present:', len(df))

entities_dict = dict()
for paper_type in ['research', 'data']:
    entities_dict[paper_type] = extract_and_save_entities(df, paper_type, nlp)

mlflow.create_experiment('raw-ner')
mlflow.set_experiment('raw-ner')
ner_method_scores('data', entities_dict, df)
ner_method_scores('research', entities_dict, df)
plt.show()