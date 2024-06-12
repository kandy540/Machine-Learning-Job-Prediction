from s3fs.core import S3FileSystem
import numpy as np
import pickle


def transform_data():
    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = 's3://'  # Insert here
    # Get data from S3 bucket as a pickle file
    raw_data = np.load(s3.open('{}/{}'.format(DIR, 'uncleaned_project_data.pkl')), allow_pickle=True)

    # EDA Process
    raw_data = raw_data.drop('index', axis=1)
    # Drop duplicates
    raw_data = raw_data.drop_duplicates()
    # Format column names in the form of lowercase words separated by '_'
    raw_data = raw_data.rename(columns=lambda header: header.lower().replace(" ", "_"))
    # Standarize Capitalizaton
    raw_data["job_title"] = raw_data.loc[:, "job_title"].str.replace("(Sr.)", "sr.")
    # Salary Estimate formatting
    raw_data['salary_estimate'] = raw_data['salary_estimate'].str.replace('K', '000')
    # remove unneccesary values
    raw_data['job_description'] = raw_data['job_description'].str.replace('\n', ' ', regex=True)
    raw_data['company_name'] = raw_data['company_name'].str.split('\n').str[0]
    # Delete the rest of the instances using a regex
    raw_data["job_title"] = raw_data.loc[:, "job_title"].str.extract('([^()]+)')
    # Replace the special characters with an empty value by defining a regex pattern
    raw_data['job_title'] = raw_data['job_title'].str.replace(r'[^a-zA-Z0-9-,/\s]', '', regex=True)
    # remove delimiters and numbers after company name
    raw_data['company_name'] = raw_data.loc[:, 'company_name'].str.replace(r"\n\d+(\.\d+)?", '', regex=True)
    # create state column
    raw_data['location_state'] = raw_data['location'].apply(lambda x: x.split(',')[-1].strip())
    # Replace rating mising values with NaN across the entire column
    raw_data['rating'].replace(-1.0, np.nan, inplace=True)
    # Standardize state abbreviations
    raw_data['company_name'] = raw_data.loc[:, 'company_name'].str.replace(r"\n\d+(\.\d+)?", '', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"California", 'CA', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"Texas", 'TX', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"Utah", 'UT', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"New Jersey", 'NJ', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"Remote", 'N/A', regex=True)
    raw_data['location_state'] = raw_data.loc[:, 'location_state'].str.replace(r"United States", 'N/A', regex=True)
    # remove missing values from headquarters
    raw_data['headquarters'] = raw_data['headquarters'].apply(lambda x: x.replace('-1', 'n/a'))
    # remove NAN values
    raw_data = raw_data.dropna()

    # Push cleaned data to S3 bucket warehouse
    DIR_wh = 's3://'  # Insert here
    with s3.open('{}/{}'.format(DIR_wh, 'clean_job_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(raw_data))
