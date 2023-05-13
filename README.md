# Automatic Linking of Data and Research Papers
This project is a follow-up to the following paper: [Deep Impact: A Study on the Impact of Data Papers and Datasets in the Humanities and Social Sciences](https://doi.org/10.3390/publications10040039).  Its aim is to automate the linking between data and research papers.

# Usage
1. Clone the repository: ``git clone https://github.com/aryanpoo/nlp-automatic-linking-proj.git``
2. Change to project directory: ``cd nlp-automatic-linking-proj``
2. Install requirements: ``pip install -r requirements.txt``
3. Create the dataset: ``python src/downloads.py``
4. Run NER method: ``python src/ner.py``