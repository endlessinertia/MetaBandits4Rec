import pandas as pd
import numpy as np

from evaluation import EvaluationProtocol
from contextual_bandit_selection import MetaBandits4Rec
from rec_algo import *


def import_ml1m_data(data_path, binarize_rating=False, user_snippet=None):

    data = pd.read_csv(data_path, sep='::', engine='python', header=None)
    data.columns = ('user_id', 'item_id', 'rating', 'timestamp')
    data.drop(columns=['timestamp'], inplace=True)

    if user_snippet:
        data = data[data['user_id'] <= user_snippet]

    data['user_id'] = data['user_id'].astype('category')
    data['item_id'] = data['item_id'].astype('category')

    if binarize_rating:
        data['rating'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

    return data

###########################################################
##################### PARAMETERS ##########################
###########################################################

data_path = 'ml-1m/ratings.dat'
rec_type = 'nmf'
ratings_type = 'explicit'
test_size = 0.95
epochs = 100
n = 1
positive_th = 4
knn = 30
min_knn = 20
num_factors = 20

###########################################################

# import data and context
data = import_ml1m_data(data_path, user_snippet=3000)
print(data)

if rec_type == 'bandit':

    algo_dict = {0: 'rand', 1: 'pop', 2: 'ucf', 3: 'nmf'} # map of the arm index to the algorithm
    context = np.eye(len(algo_dict)) # toy one-hot encoded context for each algorithm

    # create a train-test split
    rec_eval = EvaluationProtocol()
    rec_eval.train_test_split(data, test_size, split_type='user')

    # initialize bandits for algorithm meta-selection
    user_bandit_dict = dict()
    for user in rec_eval.users_to_test:
        b4r = MetaBandits4Rec(user, len(context[0]), context)
        user_bandit_dict[user] = b4r

    # initialize recommendation algorithms
    rand_model = Rand(n)
    pop_model = Pop(n)
    ucf_model = UserCF(n, knn, min_knn)
    nmf_model = MatrixFactorization(n, num_factors, ratings_type)

    # initialize report
    report = '#epoch:num_users_selection...:num_correct_retrieval...\n'
    report += '#algorithms='
    for _, algo in algo_dict.items():
        report += algo + ':'
    report = report[:-1] + '\n'
    report += '#dataset={}\n'.format(data_path)
    report += '#test_split={}:test_size={}\n'.format(test_size, rec_eval.test_data.shape[0])
    report += '##############################################################\n'


    for i in range(epochs):

        print('\n##### EPOCH {} #####\n'.format(i))
        print(rec_eval.train_data)
        print('\n')
        report += str(i) + ':'

        # fit the recommendation models
        rand_model.fit(rec_eval.train_data)
        pop_model.fit(rec_eval.train_data)
        ucf_model.fit(rec_eval.train_data)
        nmf_model.fit(rec_eval.train_data)

        algo_users_to_test_dict = {'rand': list(), 'pop': list(), 'ucf': list(), 'nmf': list()} # for each algorithm the list of users that use it

        # each user selects its best arm (algorithm) to use in this round
        for user in rec_eval.users_to_test:
            exp_rew = user_bandit_dict[user].TS_get_expected_reward()
            selected_arm = user_bandit_dict[user].select_best_arm(exp_rew)
            algo_users_to_test_dict[algo_dict[selected_arm]].append(user)
        for alg, list_users in algo_users_to_test_dict.items():
            print('List of users for {}: {}'.format(alg, list_users))
            report += str(len(list_users)) + ':'
        print('\n')

        # recommendations are done for each algorithm selected
        rand_model.rec(algo_users_to_test_dict['rand'], rec_eval.users_blacklist)
        print('Rec list for rand: {}'.format(rand_model.rec_list))
        pop_model.rec(algo_users_to_test_dict['pop'], rec_eval.users_blacklist)
        print('Rec list for pop: {}'.format(pop_model.rec_list))
        ucf_model.rec(algo_users_to_test_dict['ucf'], rec_eval.users_blacklist)
        print('Rec list for ucf: {}'.format(ucf_model.rec_list))
        nmf_model.rec(algo_users_to_test_dict['nmf'], rec_eval.users_blacklist)
        print('Rec list for nmf: {}\n'.format(nmf_model.rec_list))

        # recommendations are evaluated on the ground truth
        rand_eval_list = rec_eval.evaluate_recommendation(rand_model.rec_list, positive_th)
        print('Eval of rand recommendation list: {}'.format(rand_eval_list))
        report += str(sum(rand_eval_list.values())) + ':'
        pop_eval_list = rec_eval.evaluate_recommendation(pop_model.rec_list, positive_th)
        print('Eval of pop recommendation list: {}'.format(pop_eval_list))
        report += str(sum(pop_eval_list.values())) + ':'
        ucf_eval_list = rec_eval.evaluate_recommendation(ucf_model.rec_list, positive_th)
        print('Eval of ucf recommendation list: {}'.format(ucf_eval_list))
        report += str(sum(ucf_eval_list.values())) + ':'
        nmf_eval_list = rec_eval.evaluate_recommendation(nmf_model.rec_list, positive_th)
        print('Eval of nmf recommendation list: {}\n'.format(nmf_eval_list))
        report += str(sum(nmf_eval_list.values())) + ':'

        # update bandits parameters for each user based on the recommendation evaluation
        eval_list = dict()
        eval_list.update(rand_eval_list)
        eval_list.update(pop_eval_list)
        eval_list.update(ucf_eval_list)
        eval_list.update(nmf_eval_list)
        for user, rew in eval_list.items():
            user_bandit_dict[user].parameters_update(rew)

        report = report[:-1] + '\n'
        with open('results_report.txt', 'w') as rep_f:
            rep_f.write(report)

else:

    # create a train-test split
    rec_eval = EvaluationProtocol()
    rec_eval.train_test_split(data, test_size, split_type='user')
    users_list = set(rec_eval.train_data['user_id'].tolist())

    # initialize recommendation algorithms
    if rec_type == 'rand':
        rec_model = Rand(n)
    elif rec_type == 'pop':
        rec_model = Pop(n)
    elif rec_type == 'ucf':
        rec_model = UserCF(n, knn, min_knn)
    elif rec_type == 'nmf':
        rec_model = MatrixFactorization(n, num_factors, ratings_type)

    # initialize report
    report = '#epoch:num_users:num_hits\n'
    report += '#algorithm=' + rec_type + '\n'
    report += '#dataset={}\n'.format(data_path)
    report += '#test_split={}:test_size={}\n'.format(test_size, rec_eval.test_data.shape[0])
    report += '##############################################################\n'

    for i in range(epochs):

        print('\n##### EPOCH {} #####\n'.format(i))
        print(rec_eval.train_data)
        print('\n')
        report += str(i) + ':'

        # fit the recommendation models
        rec_model.fit(rec_eval.train_data)
        rec_model.rec(users_list, rec_eval.users_blacklist)
        print('Rec list per user: {}'.format(rec_model.rec_list))

        # recommendations are evaluated on the ground truth
        eval_list = rec_eval.evaluate_recommendation(rec_model.rec_list, positive_th)
        print('Eval of rec list per user: {}'.format(eval_list))
        report += str(sum(eval_list.values())) + ':'

        report = report[:-1] + '\n'
        with open('results_report.txt', 'w') as rep_f:
            rep_f.write(report)
