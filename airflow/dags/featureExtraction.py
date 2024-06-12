from s3fs.core import S3FileSystem
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack

def feature_extract():

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://'
    DIR_job = DIR_wh # Insert here
    # Get data from S3 bucket as a pickle file
    job_df = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_job_data.pkl')), allow_pickle=True)

    # Load pickle data locally
    #job_df = np.load('clean_job_data.pkl', allow_pickle=True)

    # Remove stopwords from job description field
    stopword_removal(job_df, input_column='job_description', new_column='job_description_sws_removed')

    #Extract seniority to new column for training
    job_df = add_seniority(job_df)

    # Create bag-of-word vectors for each job description and transfer that into a sparse array
    desc_vectors = create_word_vector(job_df, max=.4, min=.05)
    X = desc_vectors #Job description will train solely on the description vectors

    # Extract state field as a one-hot-encoded dataframe
    X_state = pd.get_dummies(job_df.location_state)
    X_state = csr_matrix(X_state)
    X_combined = hstack([X, X_state]) #Field will also train on encoded state

    # We will train two models
    Y_seniority = job_df['seniority']
    Y_sector = job_df['sector']

    X_train, X_test, y_train, y_test = train_test_split(X, Y_seniority, test_size=.2)

    # Push extracted features for seniority prediction to data warehouse
    write_to_pickle(DIR_job, 'seniority', X_test, X_train, s3, y_test, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, Y_sector, test_size=.2)
    write_to_pickle(DIR_job, 'sector', X_test, X_train, s3, y_test, y_train)


def write_to_pickle(DIR_job, subdirectory, X_test, X_train, s3, y_test, y_train):
    DIR_job = DIR_job+subdirectory
    with s3.open('{}/{}'.format(DIR_job, 'X_train_job.pkl'), 'wb') as f:
        f.write(pickle.dumps(X_train))
    with s3.open('{}/{}'.format(DIR_job, 'X_test_job.pkl'), 'wb') as f:
        f.write(pickle.dumps(X_test))
    with s3.open('{}/{}'.format(DIR_job, 'y_train_job.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_train))
    with s3.open('{}/{}'.format(DIR_job, 'y_test_job.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_test))


def stopword_removal(df_local, input_column='abstract', new_column='abstract_sws_removed'):
    """
    Removes stopwords from an input column and appends parsed text to a new column

    :param df_local: The input dataframe
    :param input_column: Column containing original text
    :param new_column:  To be created column with the parsed text
    :return: The output dataframe
    """
    # Import nltk libraries for stopword handling and tokenization
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    punctuation = ['.', ',', '?', '\'', '\"', '!']
    trigger_words = ['junior', 'Junior', 'senior', 'Senior']
    stop_words.update(punctuation)
    stop_words.update(trigger_words)

    df_local[new_column] = df_local[input_column].apply(lambda text: ' '.join(
        word for word in word_tokenize(text) if word.lower() not in stop_words))

    return df_local

def add_seniority(df_local, input_column='job_description', new_column='seniority'):
    if input_column not in df_local.columns:
        raise ValueError(f"Column '{input_column}' not found in DataFrame")

        # Function to determine the seniority based on the content of the cell

    def determine_seniority(cell_content):
        cell_content = str(cell_content).lower()  # Convert to string and make case-insensitive
        if 'junior' in cell_content:
            return 'junior'
        elif 'senior' in cell_content:
            return 'senior'
        else:
            return 'Mid'  # or return 'N/A' or any other placeholder you prefer

        # Apply the function to the specified column and create the new column

    df_local[new_column] = df_local[input_column].apply(determine_seniority)

    return df_local


def create_word_vector(df_local, column='job_description_sws_removed', max=.8, min=.05):
    from sklearn.feature_extraction.text import CountVectorizer
    import time

    start = time.time()
    vectorizer = CountVectorizer(max_df=max, min_df=min)
    words = df_local[column]
    sparse_words = vectorizer.fit_transform(words)
    print('Time taken: ', f'{(time.time() - start):.2f}', 's')
    print('Sparse matrix shape is: ', sparse_words.shape)

    return sparse_words
