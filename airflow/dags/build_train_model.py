from s3fs.core import S3FileSystem
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import tempfile

def build_train():

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_job = 's3://'                       # Insert here

    X_test_job, X_train_job, y_test_job, y_train_job = load_values(DIR_job, 'seniority', s3)

    clf_seniority, seniority_report = train_evaluate_model(X_test_job, X_train_job, y_test_job, y_train_job)

    X_test_job, X_train_job, y_test_job, y_train_job = load_values(DIR_job, 'sector', s3)

    clf_sector, sector_report = train_evaluate_model(X_test_job, X_train_job, y_test_job, y_train_job)

    # # Save models in the pickle format
    # with open('seniority_clf.pkl', 'wb') as file:
    #     pickle.dump(clf_seniority, file)
    #
    # with open('sector_clf.pkl', 'wb') as file:
    #     pickle.dump(clf_sector, file)
    #
    # Save model statistics
    with tempfile.TemporaryFile() as temp:
        # Write the accuracy to the file
        temp.write(f"Seniority classifer report:\n")
        temp.write(sector_report)
        s3.put(f"{temp}", f"{DIR_job}/seniority_stats.txt")

    with tempfile.TemporaryFile() as temp:
        # Write the classification report to the file
        temp.write("Sector classifer report:\n")
        temp.write(sector_report)
        s3.put(f"{temp}", f"{DIR_job}/sector_stats.txt")


def train_evaluate_model(X_test_job, X_train_job, y_test_job, y_train_job):
    clf = MultinomialNB()
    clf.fit(X_train_job, y_train_job)
    report = classification_report(y_test_job, clf.predict(X_test_job))
    return clf, report


def load_values(DIR_job, subdirectory, s3):
    DIR_job = DIR_job + subdirectory
    X_train_job = np.load(s3.open('{}/{}'.format(DIR_job, 'X_train_job.pkl')), allow_pickle=True)
    X_test_job = np.load(s3.open('{}/{}'.format(DIR_job, 'X_test_job.pkl')), allow_pickle=True)
    y_train_job = np.load(s3.open('{}/{}'.format(DIR_job, 'y_train_job.pkl')), allow_pickle=True)
    y_test_job = np.load(s3.open('{}/{}'.format(DIR_job, 'y_test_job.pkl')), allow_pickle=True)
    return X_test_job, X_train_job, y_test_job, y_train_job
