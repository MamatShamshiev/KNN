import numpy as np
import sklearn.neighbors as sk_knn
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds=3):
    cv_split = np.array(np.array_split(np.random.permutation(n), n_folds))
    folds = [(np.hstack(np.delete(cv_split, i, 0)), cv_split[i]) for i in range(n_folds)]
    return folds


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if cv is None:
        cv = kfold(len(X))  
    my_knn = KNNClassifier(k=k_list[-1], **kwargs)
    results = {k: np.array([]) for k in k_list}
    for fold in cv:
        my_knn.fit(X[fold[0]], y[fold[0]])
        distance_matrix, index_matrix = my_knn.find_kneighbors(X[fold[1]], return_distance=True)
        for k in k_list:
            predictions = my_knn.predict_using_distance(index_matrix[:, :k], distance_matrix[:, :k])
            accuracy = np.sum(np.equal(predictions, y[fold[1]])) / len(y[fold[1]])
            results[k] = np.append(results[k], accuracy)
    return results


def knn_cross_val_score_aug(X, y, k_list, score='accuracy', cv=None, aug_function=None,
                            aug_param=None, **kwargs):
    if cv is None:
        cv = kfold(len(X))
    my_knn = KNNClassifier(k=k_list[-1], **kwargs)
    results = {k: np.array([]) for k in k_list}
    for fold in cv:
        my_knn.fit(X[fold[0]], y[fold[0]])
        distance_matrix, index_matrix = my_knn.find_kneighbors(X[fold[1]], return_distance=True)
        for i in aug_param:
            X_train_new = np.array([aug_function(elem.reshape(28, 28), i).ravel()
                                    for elem in X[fold[0]]])
            my_knn.fit(X_train_new, y[fold[0]])
            dist, index = my_knn.find_kneighbors(X[fold[1]], return_distance=True)
            del(X_train_new)
            distance_matrix = np.hstack((distance_matrix, dist))
            index_matrix = np.hstack((index_matrix, index))
        indexes = np.argpartition(distance_matrix, range(k_list[-1]), axis=1)[:, :k_list[-1]]
        index_matrix = index_matrix[np.array([x for x in range(distance_matrix.shape[0])
                                              for i in range(k_list[-1])]),
                                    indexes.ravel()].reshape((distance_matrix.shape[0],
                                                              k_list[-1]))
        distance_matrix = distance_matrix[np.array([x for x in range(distance_matrix.shape[0])
                                                    for i in range(k_list[-1])]),
                                          indexes.ravel()].reshape((distance_matrix.shape[0],
                                                                    k_list[-1]))
        for k in k_list:
            predictions = my_knn.predict_using_distance(index_matrix[:, :k],
                                                        distance_matrix[:, :k])
            accuracy = np.sum(np.equal(predictions, y[fold[1]])) / len(y[fold[1]])
            results[k] = np.append(results[k], accuracy)
    return results


def knn_cross_val_score_aug_test(X, y, k_list, score='accuracy', cv=None, aug_function=None,
                                 aug_param=None, **kwargs):
    if cv is None:
        cv = kfold(len(X))
    my_knn = KNNClassifier(k=k_list[-1], **kwargs)
    results = {k: np.array([]) for k in k_list}
    for fold in cv:
        my_knn.fit(X[fold[0]], y[fold[0]])
        distance_matrix, index_matrix = my_knn.find_kneighbors(X[fold[1]], return_distance=True)
        for i in aug_param:
            X_test_new = np.array([aug_function(elem.reshape(28, 28), i).ravel()
                                   for elem in X[fold[1]]])
            dist, index = my_knn.find_kneighbors(X_test_new, return_distance=True)
            distance_matrix = np.hstack((distance_matrix, dist))
            index_matrix = np.hstack((index_matrix, index))
            del(dist)
            del(index)
        indexes = np.argpartition(distance_matrix, range(k_list[-1]), axis=1)[:, :k_list[-1]]
        index_matrix = index_matrix[np.array([x for x in range(distance_matrix.shape[0])
                                              for i in range(k_list[-1])]),
                                    indexes.ravel()].reshape((distance_matrix.shape[0],
                                                              k_list[-1]))
        distance_matrix = distance_matrix[np.array([x for x in range(distance_matrix.shape[0])
                                                    for i in range(k_list[-1])]),
                                          indexes.ravel()].reshape((distance_matrix.shape[0],
                                                                    k_list[-1]))
        for k in k_list:
            predictions = my_knn.predict_using_distance(index_matrix[:, :k].astype(int),
                                                        distance_matrix[:, :k])
            accuracy = np.sum(np.equal(predictions, y[fold[1]])) / len(y[fold[1]])
            results[k] = np.append(results[k], accuracy)
    return results
