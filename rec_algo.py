import random
import surprise
from collections import Counter
from operator import attrgetter


class Rand:

    def __init__(self, n):
        self.top_n = n

        self.item_list = None
        self.rec_list = {}

    def fit(self, train_data):
        self.item_list = list(set(train_data['item_id'].tolist()))

    def rec(self, users_list, users_blacklist):
        self.rec_list = {}
        for user in users_list:
            rand_item_list = random.sample(self.item_list, len(self.item_list))
            recs = [item for item in rand_item_list if item not in users_blacklist[user]]
            self.rec_list[user] = recs[:self.top_n]


class Pop:

    def __init__(self, n, pop_type='count'):
        self.top_n = n
        self.pop_type = pop_type

        self.pop_list = None
        self.rec_list = {}

    def fit(self, train_data):
        if self.pop_type == 'count':
            item_list = train_data['item_id'].tolist()
            item_count = Counter(item_list)
            self.pop_list = [item for (item, count) in item_count.most_common()]

    def rec(self, users_list, users_blacklist):
        self.rec_list = {}
        for user in users_list:
            recs = [item for item in self.pop_list if item not in users_blacklist[user]]
            self.rec_list[user] = recs[:self.top_n]


class UserCF:

    def __init__(self, n, k, min_k):
        self.top_n = n
        self.item_list = None
        self.rec_list = {}
        self.knn = surprise.prediction_algorithms.KNNWithMeans(k=k, min_k=min_k)

    def fit(self, train_data, rating_scale=(1, 5)):
        self.item_list = list(set(train_data['item_id'].tolist()))
        reader = surprise.reader.Reader(rating_scale=rating_scale)
        dataset = surprise.Dataset.load_from_df(train_data, reader)
        trainset = dataset.build_full_trainset()
        self.knn.fit(trainset)


    def rec(self, users_list, users_blacklist):
        self.rec_list = {}
        for user in users_list:
            prediction_list = [self.knn.predict(user, item) for item in self.item_list]
            sort_prediction = sorted(prediction_list, key=attrgetter('est'), reverse=True)
            recs = [item[1] for item in sort_prediction if item[1] not in users_blacklist[user]]
            self.rec_list[user] = recs[:self.top_n]


class MatrixFactorization:

    def __init__(self, n, factors, ratings_type):
        self.top_n = n
        self.item_list = None
        self.rec_list = {}
        if ratings_type == 'explicit':
            self.nmf = surprise.prediction_algorithms.matrix_factorization.SVD(n_factors=factors)
        elif ratings_type == 'implicit':
            self.nmf = surprise.prediction_algorithms.matrix_factorization.SVDpp(n_factors=factors)
        else:
            self.nmf = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=factors)

    def fit(self, train_data, rating_scale=(1, 5)):
        self.item_list = list(set(train_data['item_id'].tolist()))
        reader = surprise.reader.Reader(rating_scale=rating_scale)
        dataset = surprise.Dataset.load_from_df(train_data, reader)
        trainset = dataset.build_full_trainset()
        self.nmf.fit(trainset)


    def rec(self, users_list, users_blacklist):
        self.rec_list = {}
        for user in users_list:
            prediction_list = [self.nmf.predict(user, item) for item in self.item_list]
            sort_prediction = sorted(prediction_list, key=attrgetter('est'), reverse=True)
            recs = [item[1] for item in sort_prediction if item[1] not in users_blacklist[user]]
            self.rec_list[user] = recs[:self.top_n]

