import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
from pandas_datareader import data as pdr
import kaggle
from kaggle import KaggleApi
import zipfile


def ingest_data():
    api = KaggleApi()
    api.authenticate()
    # download dataset
    # uncleaned
    api.dataset_download_file('rashikrahmanpritom/data-science-job-posting-on-glassdoor', 'Uncleaned_DS_jobs.csv')

    # cleaned
    # api.dataset_download_file('rashikrahmanpritom/data-science-job-posting-on-glassdoor', 'Cleaned_DS_jobs.csv')

    # unzip file
    # zf = zipfile.ZipFile('data-science-job-posting-on-glassdoor.zip')
    # zf = zipfile.ZipFile('Cleaned_DS_Jobs.csv.zip')
    zf = zipfile.ZipFile('Uncleaned_DS_jobs.csv.zip')

    # load data into dataframe
    # uncleaned
    data = pd.read_csv(zf.open('../../Uncleaned_DS_jobs.csv'))

    # cleaned
    # data = pd.read_csv(zf.open('Cleaned_DS_Jobs.csv'))
    # print(data)

    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://'  # insert here
    # Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'uncleaned_project_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(data))

# ingest_data()
