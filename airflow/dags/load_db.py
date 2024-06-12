from sqlalchemy import create_engine
import numpy as np
from s3fs.core import S3FileSystem

def load_data():
    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://'  # Insert here

    # Get data from S3 bucket as a pickle file
    job_df = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_job_data.pkl')), allow_pickle=True)

    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@{endpnt}"
                           .format(user="username",
                                   pw="password",
                                   endpnt="url")) # Insert values

    engine.execute("CREATE DATABASE {db}"
                   .format(db="cca"))  # Insert pid here

    engine = create_engine("mysql+pymysql://{user}:{pw}@{endpnt}/{db}"
                           .format(user="username",
                                   pw="password",
                                   endpnt="url",
                                   db="db_name")) # Insert values


    # Insert whole DataFrame into MySQL DB
    job_df.to_sql('job_clean',  # put table name here
                   con=engine, if_exists='replace', chunksize=1000)
