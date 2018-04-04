import numpy as np
import sklearn.neighbors as sk_knn


class KNNClassifier:
    def __init__(self, k, strategy='my_own', metric='euclidean', weights=False, test_block_size=0):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise TypeError('KNNClassifier.fit: input with inconsistent numbers of samples')
        self.y = y
        if self.strategy == 'my_own':
            self.X = X
        else:
            self.knn = sk_knn.NearestNeighbors(algorithm=self.strategy,
                                               n_neighbors=self.k, metric=self.metric)
            self.knn.fit(X, y)

    def find_kneighbors(self, X, return_distance):
        if self.test_block_size == 0 or self.test_block_size > len(X):
            test_block_size = len(X)
        else:
            test_block_size = self.test_block_size
        dist_matrix = np.empty((0, self.k))
        index_matrix = np.empty((0, self.k))
        if self.strategy == 'my_own':
            train_norms = np.sum(self.X ** 2, axis=1)
            if self.metric == 'euclidean':
                for i in range(len(X) // test_block_size):
                    split = X[i*test_block_size:(i+1)*test_block_size]
                    test_norms = np.sum(split ** 2, axis=1)
                    dist = np.subtract(np.add(test_norms[:, np.newaxis],
                                              train_norms[np.newaxis, :]),
                                       np.dot(split, self.X.T) * 2) ** 0.5
                    indexes = np.argpartition(dist, range(self.k), axis=1)[:, :self.k]
                    index_matrix = np.vstack((index_matrix, indexes))
                    dist = dist[np.array([x for x in range(test_block_size)
                                          for i in range(self.k)]),
                                indexes.ravel()].reshape((test_block_size, self.k))
                    dist_matrix = np.vstack((dist_matrix, dist))
                if len(X) % test_block_size != 0:
                    split = X[(len(X) // test_block_size)*test_block_size:]
                    test_norms = np.sum(split ** 2, axis=1)
                    dist = np.subtract(np.add(test_norms[:, np.newaxis],
                                              train_norms[np.newaxis, :]),
                                       np.dot(split, self.X.T) * 2) ** 0.5
            elif self.metric == 'cosine':
                train_norms = train_norms ** 0.5
                for i in range(len(X) // test_block_size):
                    split = X[i*test_block_size:(i+1)*test_block_size]
                    test_norms = np.sum(split ** 2, axis=1) ** 0.5
                    norms = np.dot(test_norms[:, np.newaxis], train_norms[np.newaxis, :])
                    dist = np.subtract(np.ones((test_block_size, self.X.shape[0])),
                                       np.divide(np.dot(split, self.X.T), norms))
                    indexes = np.argpartition(dist, range(self.k), axis=1)[:, :self.k]
                    index_matrix = np.vstack((index_matrix, indexes))
                    dist = dist[np.array([x for x in range(test_block_size)
                                          for i in range(self.k)]),
                                indexes.ravel()].reshape((test_block_size, self.k))
                    dist_matrix = np.vstack((dist_matrix, dist))
                if len(X) % test_block_size != 0:
                    split = X[(len(X) // test_block_size)*test_block_size:]
                    test_norms = np.sum(split ** 2, axis=1) ** 0.5
                    norms = np.dot(test_norms[:, np.newaxis], train_norms[np.newaxis, :])
                    dist = np.subtract(np.ones((split.shape[0], self.X.shape[0])),
                                       np.divide(np.dot(split, self.X.T), norms))
            else:
                raise TypeError('KNNClassifier.find_kneighbors: wrong metric type!')
            if len(X) % test_block_size != 0:
                indexes = np.argpartition(dist, range(self.k), axis=1)[:, :self.k]
                index_matrix = np.vstack((index_matrix, indexes))
                dist = dist[np.array([x for x in range(dist.shape[0]) for i in range(self.k)]),
                            indexes.ravel()].reshape((dist.shape[0], self.k))
                dist_matrix = np.vstack((dist_matrix, dist))

            if return_distance is True:
                return (dist_matrix, index_matrix.astype(int))
            elif return_distance is False:
                return index_matrix.astype(int)
            else:
                raise TypeError('KNNClassifier.find_kneighbors: return_distance must be a bool!')
        else:
            return self.knn.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        if self.weights is True:
            dist_matrix, index_matrix = self.find_kneighbors(X, return_distance=True)
            weights = np.divide(np.ones(dist_matrix.shape), np.add(np.full(dist_matrix.shape, 1e-5),
                                                                   dist_matrix))
        elif self.weights is False:
            index_matrix = self.find_kneighbors(X, return_distance=False)
            weights = np.ones((X.shape[0], self.k))
        else:
            raise TypeError('KNNClassifier.predict: weights must be a bool!')
        labels = self.y[index_matrix.ravel()].reshape(index_matrix.shape)
        max_sums = np.zeros(weights.shape[0])
        max_labels = np.zeros(X.shape[0])
        for label in np.unique(labels):
            cur_sum = np.sum(np.where(labels == label, weights, np.zeros(weights.shape)), axis=1)
            max_sums = np.maximum(cur_sum, max_sums)
            max_labels[max_sums == cur_sum] = label
        return max_labels.astype(int)

    def predict_using_distance(self, index_matrix, dist_matrix):
        if self.weights is True:
            weights = np.divide(np.ones(dist_matrix.shape), np.add(np.full(dist_matrix.shape, 1e-5),
                                                                   dist_matrix))
        elif self.weights is False:
            weights = np.ones(dist_matrix.shape)
        else:
            raise TypeError('KNNClassifier.predict_using_distance: weights must be a bool!')
        labels = self.y[index_matrix.ravel()].reshape(index_matrix.shape)
        max_sums = np.zeros(weights.shape[0])
        max_labels = np.zeros(weights.shape[0])
        for label in np.unique(labels):
            cur_sum = np.sum(np.where(labels == label, weights, np.zeros(weights.shape)), axis=1)
            max_sums = np.maximum(cur_sum, max_sums)
            max_labels[max_sums == cur_sum] = label
        return max_labels.astype(int)
