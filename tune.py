import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd
import numpy as np


def LightGBMCallback(env):
    """Assumes that `valid_0` is the target validation score."""
    _, metric, score, _ = env.evaluation_result_list[0]
    tune.report(**{metric: score})


def train_breast_cancer(config):
    pv_pb_df_mn = pd.read_csv(
        'data aggregation/pv_pb_df_dp.csv'
    )
    norm = [
        'time', 'count_neshan', 'count_scat_o', 'count_avl_en_o',
        'count_avl_ex_o', 'count_anpr_o',
        'karmnd_dr_mhl_shghl_o', 'veh_own_o', 'n_bussi_unit_o', 'park_area_o', 'area_o',
        'office_land_use_o', 'n_office_o', 'commercial_unit_o', 'n_commercial_o', 'schl_o', 'count_scat_d', 'count_avl_en_d', 'count_avl_ex_d',
        'count_anpr_d', 'karmnd_dr_mhl_shghl_d',
        'veh_own_d', 'n_bussi_unit_d', 'park_area_d', 'area_d', 'office_land_use_d', 'n_office_d', 'commercial_unit_d',
        'n_commercial_d', 'schl_d', 'count_pv_pb'
    ]

    pv_pb_df_mn[norm] = pv_pb_df_mn[norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    y = pv_pb_df_mn['count_pv_pb']
    # with avl
    X = pv_pb_df_mn[['time', 'count_neshan', 'count_scat_o', 'count_avl_en_o',
                     'count_avl_ex_o', 'count_anpr_o',
                     'karmnd_dr_mhl_shghl_o', 'veh_own_o', 'n_bussi_unit_o', 'park_area_o', 'area_o',
                     'office_land_use_o', 'n_office_o', 'commercial_unit_o', 'n_commercial_o', 'schl_o', 'count_scat_d', 'count_avl_en_d', 'count_avl_ex_d',
                     'count_anpr_d', 'karmnd_dr_mhl_shghl_d',
                     'veh_own_d', 'n_bussi_unit_d', 'park_area_d', 'area_d', 'office_land_use_d', 'n_office_d', 'commercial_unit_d',
                     'n_commercial_d', 'schl_d']]

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.25)
    train_set = lgb.Dataset(train_x, label=train_y)
    test_set = lgb.Dataset(test_x, label=test_y)
    gbm = lgb.train(
        config,
        train_set,
        valid_sets=[test_set],
        verbose_eval=False,
        callbacks=[LightGBMCallback], )
    preds = gbm.predict(test_x)
    tune.report(
        mean_accuracy=sklearn.metrics.r2_score(test_y, preds),
        done=True)


if __name__ == "__main__":
    config = {
        "objective": "regression",
        "metric": "l2",
        "num_iterations": 1,
        "verbose": -1,
        "bagging_fraction": tune.grid_search([0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "feature_fraction": tune.grid_search([0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "max_bin": tune.randint(80, 1000),
        "max_depth": tune.choice(
            [
                -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ]
        ),
        "boosting_type": tune.grid_search(["gbdt", "dart", "goss", "rf"]),
        "num_leaves": tune.randint(80, 100),
        "learning_rate": tune.grid_search(
            [
                0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.0001]
        )
    }

    analysis = tune.run(
        train_breast_cancer,
        metric="l2",
        mode="max",
        config=config,
        num_samples=10,
        scheduler=ASHAScheduler(max_t=1))

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best resultss found were: ", analysis.best_result)

# Get a dataframe for analyzing trial results.
df = analysis.results_df
