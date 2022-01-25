import numpy as np


def get_fairness_measures(test_dataset, preds):
    group_label = test_dataset.adv_mask #f['group_label']
    Y_test = test_dataset.Y #self.data_sets.test.labels

    index_male_test = np.where(group_label == 0)[0].astype(np.int32)
    index_female_test = np.where(group_label == 1)[0].astype(np.int32)

    poi_test_hat_one = np.where(preds == 1)[0].astype(np.int32)
    poi_test_hat_zero = np.where(preds == 0)[0].astype(np.int32)

    poi_test_y_one_hat_one =  (np.where(np.logical_and(preds == 1, Y_test==1))[0].astype(np.int32).shape[0]) / Y_test.shape[0]
    poi_test_y_one_hat_zero = (np.where(np.logical_and(preds == 0, Y_test==1))[0].astype(np.int32).shape[0])/Y_test.shape[0]

    test_female_one_prediction = np.where(preds[index_female_test] == 1)[0].astype(np.int32)
    test_female_zero_prediction = np.where(preds[index_female_test] == 0)[0].astype(np.int32)
    test_male_one_prediction = np.where(preds[index_male_test] == 1)[0].astype(np.int32)
    test_male_zero_prediction = np.where(preds[index_male_test] == 0)[0].astype(np.int32)
    print("VRWMIARA - SPD" + str (   abs( (test_female_one_prediction.shape[0]/index_female_test.shape[0]) - (test_male_one_prediction.shape[0]/index_male_test.shape[0])  )     ))

    a_female_test = (test_female_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_one_hat_one
    a_male_test = (test_male_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_one_hat_one

    b_female_test = (test_female_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_one_hat_zero
    b_male_test = (test_male_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_one_hat_zero

    print("VRWMIARA - EOD" + str ( abs( (a_female_test/(a_female_test+b_female_test)) - (a_male_test/(a_male_test+b_male_test)) )) )
