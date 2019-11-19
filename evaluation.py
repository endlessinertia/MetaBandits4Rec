import numpy as np
import pandas as pd

class EvaluationProtocol:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.users_blacklist = None
        self.users_to_test = None

    def train_test_split(self, dataset, test_ratio=0.25, split_type='random'):
        if split_type == 'random':
            temp_data = dataset.sample(frac=1)
            msk = np.random.rand(len(temp_data)) < test_ratio
            self.train_data = temp_data[~msk]
            self.test_data = temp_data[msk]
        elif split_type == 'user':
            temp_train_df_list = list()
            temp_test_df_list = list()
            users_group_dataset = dataset.groupby('user_id')
            for user_id, user_df in users_group_dataset:
                temp_user_df = user_df.sample(frac=1)
                msk = np.random.rand(temp_user_df.shape[0]) < test_ratio
                temp_train_df_list.append(temp_user_df[~msk])
                temp_test_df_list.append(temp_user_df[msk])
            self.train_data = pd.concat(temp_train_df_list, ignore_index=True)
            self.test_data = pd.concat(temp_test_df_list, ignore_index=True)

        self.users_to_test = list(set(self.train_data['user_id']) & set(self.test_data['user_id']))
        self.initialize_users_blacklist()

    def initialize_users_blacklist(self):
        self.users_blacklist = dict()
        for user_id in self.users_to_test:
            blacklist = self.train_data[self.train_data['user_id'] == user_id]['item_id'].tolist()
            self.users_blacklist[user_id] = blacklist

    def update_users_blacklist(self, user_id, item_list):
        user_blacklist = self.users_blacklist[user_id]
        user_blacklist.extend(item_list)
        self.users_blacklist[user_id] = user_blacklist

    def consume_items(self, user_id, item_list, ratings):
        self.update_users_blacklist(user_id, item_list)
        extended_train_data = pd.DataFrame({'user_id': user_id, 'item_id': item_list, 'rating': ratings})
        temp_train = self.train_data
        self.train_data = temp_train.append(extended_train_data, ignore_index=True)

    def consume_items_df(self, consumed_items_df):
        temp_train = self.train_data
        self.train_data = temp_train.append(consumed_items_df, ignore_index=True)

    def evaluate_recommendation(self, rec_list, positive_threshold, consume_negative=True):
        hit_count_dict = {}

        for user_id, user_rec in rec_list.items():
            user_ground_truth = self.test_data[self.test_data['user_id'] == user_id]
            self.update_users_blacklist(user_id, user_rec)

            positive_items = user_ground_truth[user_ground_truth['rating'] >= positive_threshold]['item_id'].tolist()
            positive_recall_list = list(set(user_rec) & set(positive_items))
            positive_rec_items = user_ground_truth[user_ground_truth['item_id'].isin(positive_recall_list)]
            self.consume_items_df(positive_rec_items)

            hit_count_dict[user_id] = len(positive_recall_list)

            if consume_negative:
                negative_items = user_ground_truth[user_ground_truth['rating'] < positive_threshold]['item_id'].tolist()
                negative_recall_list = list(set(user_rec) & set(negative_items))
                negative_rec_items = user_ground_truth[user_ground_truth['item_id'].isin(negative_recall_list)]
                self.consume_items_df(negative_rec_items)

        return hit_count_dict

    # def evaluate_recommendation(self, rec_list, positive_threshold=1, consume_negative=None):
    #
    #     hit_count_dict = {}
    #
    #     for user_id, user_rec in rec_list.items():
    #         user_ground_truth = self.test_data[self.test_data['user_id'] == user_id]
    #         positive_items = user_ground_truth[user_ground_truth['rating'] >= positive_threshold]['item_id'].tolist()
    #         positive_recall_list = list(set(user_rec) & set(positive_items))
    #         self.consume_items(user_id, positive_recall_list, 1)
    #         hit_count_dict[user_id] = len(positive_recall_list)
    #
    #         if consume_negative == 'explicit':
    #             negative_items = user_ground_truth[user_ground_truth['rating'] < positive_threshold]['item_id'].tolist()
    #             negative_recall_list = list(set(user_rec) & set(negative_items))
    #             self.consume_items(user_id, negative_recall_list, 0)
    #
    #         elif consume_negative == 'implicit':
    #             implicit_negative_items = list(set(user_rec) - set(positive_recall_list))
    #             self.consume_items(user_id, implicit_negative_items, 0)
    #
    #     return hit_count_dict
