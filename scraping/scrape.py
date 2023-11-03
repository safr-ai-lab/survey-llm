import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def clean_soup(string):
    string = re.sub("\<.*?\>","",string.lower()).replace('\n','')
    string = re.sub(' +', ' ',string)
    return string

from_date = '2023-08-01'

results_df = pd.DataFrame()

keywords = {'memorization': ['llm', 'language model', 'memorization', 'memorized', 'privacy', 'eidetic', 'extractibility', 'forgetting'],
            'privacy attack': ['llm', 'language model', 'privacy', 'attack', 'membership inference', 'shadow model', 'threshold', 'extraction'],
            'inference attack': ['llm', 'language model', 'privacy', 'attack', 'membership inference', 'shadow model', 'threshold', 'extraction'],
            'extraction attack': ['llm', 'language model', 'privacy', 'attack', 'membership inference', 'shadow model', 'threshold', 'extraction'],
            'private model': ['llm', 'language model', 'privacy preserving', 'private', 'differential privacy', 'federated learning', 'federated averaging'],
            'privacy preserving': ['llm', 'language model', 'privacy preserving', 'private', 'differential privacy', 'federated learning', 'federated averaging'],
            'unlearning': ['llm', 'language model', 'unlearning', 'leave one out', 'sisa'],
            'copyright': ['llm', 'language model', 'copyright', 'fair use', 'near access', 'unlearning', 'novelty']}

for category in keywords:
    print(f'searching {category}')
    URL = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=language+model&terms-0-field=abstract&terms-1-operator=AND&terms-1-term={category}&terms-1-field=abstract&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={from_date}&date-to_date=&date-date_type=submitted_date_first&abstracts=show&size=200&order=-announced_date_first"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all('li', attrs={'class':'arxiv-result'})
    print(f'{len(results)} results found')
    for result in results:
        title = clean_soup(str(result.find('p', attrs={'class': 'title is-5 mathjax'})))
        authors = list(clean_soup(str(result.find('p', attrs={'class': 'authors'}))).replace('authors:','').split(', '))
        abstract = clean_soup(str(result.find('span',attrs={'class':'abstract-full has-text-grey-dark mathjax'})))
        relevance = 0
        for k in keywords[category]:
            relevance += abstract.count(k)
        if category in ['inference attack','extraction attack', 'privacy attack']:
            cat = 'attack'
        elif category in ['privacy']:
            cat = 'private'
        else:
            cat = category
        data = pd.DataFrame({"title": [title],
                             "authors": authors[0],
                             "category": [cat],
                             "relevance": [relevance],
                             "abstract": [abstract]})
        results_df = results_df.append(data,ignore_index=True)

results_df = results_df[results_df.relevance > 4]
results_df = results_df.sort_values('relevance', ascending=False).drop_duplicates(subset='title', keep='first')
results_df.sort_values('relevance', ascending=False).to_csv(f'results/results_{from_date}.csv')