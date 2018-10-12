from sklearn.ensemble import RandomForestClassifier

def english_classifier(vectorised_normalised_data, data_answers):
    """
    Fits normalised data to LinearSVC

    vectorisedNormalisedData is a 2D list represented as follows
    [[A], [B], ... , [C]]
    Where A, B, C are training data for documents A, B, C
    A, B, C are represented as:
    [x, y, ..., z]
    Where x, y, z are features of document A, B, C
    dataAnswers is represented as follows:
    [a, b, ..., c]
    Where a, b, c are the authors of documents A, B, C and are matched via the index
    """
    # LinearSVC is used as a multiclass, using a one vs all approach
    clf = RandomForestClassifier(verbose=True, n_estimators=100)

    # Fitting training data with answers
    clf.fit(vectorised_normalised_data, data_answers)
    return clf
