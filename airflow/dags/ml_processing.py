def evaluate_classifier(df_local, max=.9, min=.01, n_folds=5, nn=10):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    import time
    import pandas as pd

    df_local = stopword_removal(df_local, input_column='job_description', new_column='job_description_sws_removed')
    df_local = add_seniority(df_local)
    desc_vectors = create_word_vector(df_local, max=max, min=min)
    sectors = df_local['sector']

    X = desc_vectors
    X_state = pd.get_dummies(df_local.location_state)
    X_state = csr_matrix(X_state)
    X_combined = hstack([X, X_state])

    Y_seniority = df_local['seniority']
    Y_sector = df_local['sector']

    clf = MultinomialNB()
    knn = KNeighborsClassifier(n_neighbors=nn)
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 100, 100, 100), max_iter=500, activation='relu', solver='adam',
                        random_state=42)

    print("Learning/evaluating seniority")
    start = time.time()
    scores = cross_val_score(clf, X, Y_seniority, cv=n_folds, scoring='accuracy')
    print("Multinomial Naive Bayesian Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")
    start = time.time()
    scores = cross_val_score(knn, X, Y_seniority, cv=n_folds, scoring='accuracy')
    print("KNN Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")
    start = time.time()
    scores = cross_val_score(mlp, X, Y_seniority, cv=n_folds, scoring='accuracy')
    print("Neural Network Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")

    print("")

    print("Learning/evaluating sector")
    start = time.time()
    scores = cross_val_score(clf, X_combined, Y_sector, cv=n_folds, scoring='accuracy')
    print("Naive Bayesian Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")
    start = time.time()
    scores = cross_val_score(knn, X_combined, Y_sector, cv=n_folds, scoring='accuracy')
    print("KNN Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")
    start = time.time()
    scores = cross_val_score(mlp, X_combined, Y_sector, cv=n_folds, scoring='accuracy')
    print("Neural Network Average accuracy:", scores.mean(), "Time:", f"{(time.time() - start):.2f}")

    return None
