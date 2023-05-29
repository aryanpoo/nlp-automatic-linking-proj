from os import path
import time
import json
from glob import glob
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import mimetypes

import requests
import pandas as pd
from selenium import webdriver

browser = None

def resolve_doi(doi):
    """ Resolve a DOI to the corresponding URL.

    Adapted from: https://stackoverflow.com/a/72022661
    """
    if '://' in doi:
        return doi
    url = 'https://doi.org/' + doi
    res = requests.get(url, allow_redirects=True)
    return res.url
# print(resolve_doi('10.5334/johd.1'))


def download_by_selenium(url, fn):
    """(depreciated) Download a file (like a pdf file) using Selenium."""
    global browser
    if browser is None:
        options = webdriver.ChromeOptions()
        options.add_experimental_option('prefs', {
            'download.default_directory': path.dirname(fn),
            'download.prompt_for_download': False,
            'download.directory_upgrade': True,
            'plugins.always_open_pdf_externally': True
        })
        browser = webdriver.Chrome(options=options)
        # content_type = requests.head(url).headers['Content-Type']
        # if content_type == 'text/html':
        if not fn.endswith('.pdf'):
            html_source_code = browser.execute_script('return document.body.innerHTML;')
            with open(fn, 'bw') as of:
                of.write(html_source_code)

    browser.get(url)


def download_johd_paper(url, fn, method='selenium', force_download=False):
    """ Download a paper from Journal of Open Humanities Data.
    
    Args:
        url: Url of a johd article.
        fn: Full file path without extensions.
        method: Download method. Currently only 'selenium' method is supported.
        force_download: Force re-downloading the file even it already exists.
    """
    content_fn = fn + '.html'
    abstract_fn = fn + '.abstract.txt'
    if not force_download and path.exists(content_fn) and path.exists(abstract_fn):
        print('File exists, ignored. (use force_download=True to re-download)')
        return

    assert method in ['selenium', 'api']

    if method == 'selenium':
        global browser
        if browser is None:
            browser = webdriver.Chrome()
        browser.get(url)
        get_article_js = """return function() {
            document.getElementsByClassName('article-references')[0].remove();
            return document.getElementById('xml-article').innerHTML;
        }()"""
        get_abstract_js = """return function() {
            let abstractElement = document.querySelector('div[data-testid="abstract"]');
            return abstractElement.querySelector('p').textContent;
        }()"""
        for i in range(100):
            browser.execute_script("window.scrollBy(0,250)")
        time.sleep(2.5)
        abstract = browser.execute_script(get_abstract_js)
        content = browser.execute_script(get_article_js)
        
        with open(content_fn, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(abstract_fn, 'w', encoding='utf-8') as f:
            f.write(abstract)
    else:
        raise NotImplementedError()


def get_openalex_file_url(doi):
    """Get url of a paper's file from OpenAlex database."""
    api_url = 'https://api.openalex.org/works/https://doi.org/' + doi

    with urllib.request.urlopen(api_url) as f:
        data = json.load(f)
        oa = data['open_access']
        return oa['is_oa'], oa['oa_status'], oa['oa_url']


def download_by_openalex(doi, fn, force_download=False):
    """Download a paper based on data in OpenAlex database."""
    try:
        is_oa, oa_status, oa_url = get_openalex_file_url(doi)

        print(is_oa, oa_status, oa_url)
        if oa_url is None:
            print('WARNING: CLOSED ARTICLE')
        else:
            return direct_download(oa_url, fn, force_download)
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        print(e)

def direct_download(url, fn, force_download):
    req = urllib.request.Request(
                url,
                data=None,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                }
            )

    with urllib.request.urlopen(req) as resp:
        content_type = resp.info().get_content_type()
        ext = mimetypes.guess_extension(content_type)
        if not fn.endswith(ext):
            fn += ext
        if not force_download and path.exists(fn):
            print('File exists, ignored. (use force_download=True to re-download)')
            return
        with open(fn, 'wb') as of:
            of.write(resp.read())
        print('Done!')


def download_paper(doi, url, dest_dir, force_download=False):
    """ Download a paper with a specified DOI.

    The uses site specific methods to find and extract the paper's
    text. The text will be saved in the directory specified by `dest_dir'.
    The name of the file will be determined by the paper's DOI (`/' is
    replaced with an underscore)
    """
    print(f'Downloading {doi} ({url})')

    fn = get_fn_of_doi(doi, dest_dir)

    if url.startswith('https://openhumanitiesdata.metajnl.com/'):
        download_johd_paper(url, fn, method='selenium', force_download=force_download)
    elif '://' in doi:
        direct_download(doi, fn, force_download)
    else:
        download_by_openalex(doi, fn, force_download)


def get_fn_of_doi(doi, dest_dir, ext='', already_exists=False):
    """ Get filename and path from a DOI.

    Args:
        ext: File extension with the leading dot (such as '.html').
        already_exists: A Boolean flag that if true, assumes the file exists and finds the extension based on the saved file.
    """
    assert not(ext != '' and already_exists)  
    if '://' in doi:
        d = urlparse(doi)
        fn = path.basename(d.path)
    else:
        fn = doi.replace('/', '_') + ext
    
    fn = path.join(dest_dir, fn)
    if not already_exists:
        return fn
    else:
        a = list(glob(f'{fn}*'))
        a = [x for x in a if '.abstract.' not in x]
        return a[0] if len(a) > 0 else None




BASE_DATA_DIR = 'data'
DATA_PAPERS_DIR = path.join(BASE_DATA_DIR, 'data_papers')
RESEARCH_PAPERS_DIR = path.join(BASE_DATA_DIR, 'research_papers')
Path(DATA_PAPERS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESEARCH_PAPERS_DIR).mkdir(parents=True, exist_ok=True)
DATASET_FN = path.join(BASE_DATA_DIR, 'research_datapapers-links-johd-processed.csv')
DATASET_RAW_FN = path.join(BASE_DATA_DIR, 'research_datapapers-links-johd.csv')
DATASET_RAW_URL = 'https://raw.githubusercontent.com/npedrazzini/DataPapersAnalysis/main/curated_inputs/research_datapapers-links-johd.csv'.replace('\n', '')

def make_doi_url_dataset(force_download=False):
    """Add resolved doi urls to the raw project dataset.
    
    The dataset is at https://raw.githubusercontent.com/npedrazzini/DataPapersAnalysis/main/curated_inputs/research_datapapers-links-johd.csv
    """
    
    if not force_download and path.exists(DATASET_FN):
        return

    urllib.request.urlretrieve(DATASET_RAW_URL, DATASET_RAW_FN)
    df = pd.read_csv(DATASET_RAW_FN)
    df = df.rename(columns={'DOI_data_paper': 'data_paper_doi',
                            'DOI': 'research_paper_doi'})
    # Uncomment for a quick test
    # df = df.loc[:3,:]
    df['data_paper_url'] = ''
    df['research_paper_url'] = ''
    df['research_paper_file_url'] = ''

    print('\nResolving doi of data papers ...')
    for i, doi in enumerate(df['data_paper_doi']):
        print(f'{i}/{len(df)} {doi} ', end='')
        url = resolve_doi(doi)
        print(url)
        df.loc[i, 'data_paper_url'] = url

    print('\nResolving doi of research papers ...')
    for i, doi in enumerate(df['research_paper_doi']):
        print(f'{i}/{len(df)} {doi} ', end='')
        url = resolve_doi(doi)
        print(url)
        df.loc[i, 'research_paper_url'] = url

    print('\nResolving file url of research papers ...')
    for i, doi in enumerate(df['research_paper_doi']):
        print(f'{i}/{len(df)} {doi} ', end='')
        try:
            is_oa, oa_status, oa_url = get_openalex_file_url(doi)
            print(is_oa, oa_status, oa_url)
            df.loc[i, 'research_paper_file_url'] = oa_url
        except urllib.error.HTTPError as e:
            print(e)

    df.to_csv(DATASET_FN)


if __name__ == '__main__':
    make_doi_url_dataset()
    df = pd.read_csv(DATASET_FN, index_col=0)
   
    print('Downloading papers ...')
    print('''# FIXME: currently, the script does not download the following files (correctly), please download them manually.
## 4/29 10.22230/src.2014v5n4a187 Downloading 10.22230/src.2014v5n4a187 (https://src-online.ca/index.php/src/article/view/187)
## 10/29 10.1515/opli-2016-0026 Downloading 10.1515/opli-2016-0026 (https://www.degruyter.com/document/doi/10.1515/opli-2016-0026/html)
## 11/29 10.12688/openreseurope.13843.3 Downloading 10.12688/openreseurope.13843.3 (https://open-research-europe.ec.europa.eu/articles/1-79/v3)
## 21/29 10.1515/lingvan-2021-0053 Downloading 10.1515/lingvan-2021-0053 (https://www.degruyter.com/document/doi/10.1515/lingvan-2021-0053/html)
## 22/29 http://ceur-ws.org/Vol-2723/short26.pdf HTTP Error 404: NOT FOUND
## 23/29 http://ceur-ws.org/Vol-2723/long7.pdf HTTP Error 404: NOT FOUND
## 26/29 10.6084/m9.figshare.14743044.v2 Downloading 10.6084/m9.figshare.14743044.v2 (https://figshare.com/articles/conference_contribution/Content_Reconstruction_of_Parliamentary_Questions_-_Combining_Metadata_with_an_OCR_Process/14743044/2)
## 27/29 10.1080/23273798.2018.1552007 Downloading 10.1080/23273798.2018.1552007 (https://www.tandfonline.com/doi/full/10.1080/23273798.2018.1552007)
''')
    
    # Downloading papers
    for index, row in df.iterrows():
        doi = row['data_paper_doi']
        print(f'\n{index+1}/{len(df)} {doi} ', end='')
        download_paper(doi, row['data_paper_url'], DATA_PAPERS_DIR)

    for index, row in df.iterrows():
        doi = row['research_paper_doi']
        print(f'\n{index+1}/{len(df)} {doi} ', end='')
        download_paper(doi, row['research_paper_url'], RESEARCH_PAPERS_DIR)

    # Add (or update) df with file paths
    for row_i, row in df.iterrows():
        doi = row['data_paper_doi']
        data_paper_fn = get_fn_of_doi(doi, DATA_PAPERS_DIR, already_exists=True)
        df.loc[row_i, 'data_paper_fn'] = data_paper_fn
        
        doi = row['research_paper_doi']
        research_paper_fn = get_fn_of_doi(doi, RESEARCH_PAPERS_DIR, already_exists=True)
        df.loc[row_i, 'research_paper_fn'] = research_paper_fn
    
    df.to_csv(DATASET_FN)
