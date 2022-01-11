# -*- coding: utf-8 -*-
import os
import click
import MeCab
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

vocab_file = 'vocab.pkl'
corpus_file = 'corpus.pkl'
dataset_dir = Path(__file__).resolve().parents[2] / 'data/processed'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath)

    df.drop_duplicates(inplace=True)  # 重複削除
    df = df[df['title'].str.contains('ジャルジャルのネタのタネ')]  # 稀に存在すネタ以外の動画を削除

    # タイトルの抽出
    df['title'] = df['title'].str.replace('ジャルジャルのネタのタネ', '')
    df['title'] = df['title'].str.replace('【JARUJARUTOWER】', '')
    df['title'] = df['title'].str[1:-1]

    mecab_wakati = MeCab.Tagger('-Owakati')
    df['title'] = df['title'].apply(lambda x: mecab_wakati.parse(x))
    df['title'] = df['title'].str.replace('\n', '<eos>')
    words = ' '.join(df['title'].values).split()

    id_to_word = {}
    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    corpus_path = dataset_dir / corpus_file
    vocab_path = dataset_dir / vocab_file
    
    np.save(corpus_path, corpus)
    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
