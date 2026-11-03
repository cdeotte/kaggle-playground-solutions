#!/usr/bin/env python
"""Public notebook "xgboost baseline" (version 6) run on our 5 folds.
https://www.kaggle.com/code/berkayzkan/xgboost-baseline

Features (the notebook's, label-free, built once on all train rows and once on all test rows): 47 engineered columns
added to the 14 competition columns: tyre-age / race-progress interactions, lap time / degradation / lap-time delta per
tyre lap or race lap, position and position-change signals, race-phase / stint / pit-stop flags, tyre-age quartile flags
(quartiles of the frame the features are built on), compound flags and compound x tyre-life / progress, recent-season
and testing flags. Preprocessing per fold (scikit-learn pipeline): median imputation of the numeric columns,
most-frequent imputation + ordinal encoding (unknown -> -1) of Driver, Compound and Race.
Model: XGBoost (GPU hist) with the best parameters of the notebook's 100-trial Optuna search (trial 74): 5,000 trees,
max_depth 10, lr 0.00484, subsample 0.911, colsample_bytree 0.703, min_child_weight 8.16, gamma 1.07, alpha 0.697,
lambda 3.87, scale_pos_weight 1 (the notebook computes it on its class-balanced search sample), seed 42, no early
stopping. Test = mean of the 5 fold models; OOF and test are clipped to [1e-7, 1 - 1e-7].
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

FOLDER = 'berkayzkan_xgb'
PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'n_jobs': -1,
    'scale_pos_weight': 1.0,
    # best parameters of the notebook's Optuna search (trial 74)
    'n_estimators': 5000,
    'max_depth': 10,
    'learning_rate': 0.004838209265058836,
    'subsample': 0.9107578435231197,
    'colsample_bytree': 0.7033020439199974,
    'min_child_weight': 8.15806322496324,
    'gamma': 1.0679609813781614,
    'reg_alpha': 0.6971966571498458,
    'reg_lambda': 3.8737692636495855,
}


def add_features(df):
    out = df.copy()
    eps = 1e-6

    # Explicit proxy for how many laps have passed since the latest pit/tyre change.
    out['LapsSincePit'] = out['TyreLife']
    out['LapsSincePit_Ratio'] = out['LapsSincePit'] / (out['LapNumber'] + 1)
    out['LapsSincePit_x_Progress'] = out['LapsSincePit'] * out['RaceProgress']
    out['LapsRemaining_x_LapsSincePit'] = (1 - out['RaceProgress']) * out['LapsSincePit']

    # Core tyre/race-progress interactions.
    out['Lap_TyreLife_Ratio'] = out['LapNumber'] / (out['TyreLife'] + 1)
    out['Progress_x_TyreLife'] = out['RaceProgress'] * out['TyreLife']
    out['RemainingRace'] = 1 - out['RaceProgress']
    out['RemainingRace_x_TyreLife'] = out['RemainingRace'] * out['TyreLife']
    out['TyreLife_per_Progress'] = out['TyreLife'] / (out['RaceProgress'] + 0.05)

    # Pace and degradation normalized by tyre age or race progress.
    out['LapTime_per_TyreLife'] = out['LapTime (s)'] / (out['TyreLife'] + 1)
    out['Degradation_per_TyreLife'] = out['Cumulative_Degradation'] / (out['TyreLife'] + 1)
    out['LapDelta_per_TyreLife'] = out['LapTime_Delta'] / (out['TyreLife'] + 1)
    out['Degradation_per_Lap'] = out['Cumulative_Degradation'] / (out['LapNumber'] + 1)
    out['LapTime_x_RaceProgress'] = out['LapTime (s)'] * out['RaceProgress']
    out['LapDelta_x_TyreLife'] = out['LapTime_Delta'] * out['TyreLife']

    # Position and race-state signals.
    out['Abs_Position_Change'] = out['Position_Change'].abs()
    out['Position_x_Progress'] = out['Position'] * out['RaceProgress']
    out['Position_x_TyreLife'] = out['Position'] * out['TyreLife']
    out['IsTop5'] = (out['Position'] <= 5).astype(int)
    out['IsTop10'] = (out['Position'] <= 10).astype(int)
    out['IsBackMarker'] = (out['Position'] >= 16).astype(int)
    out['GainedPosition'] = (out['Position_Change'] < 0).astype(int)
    out['LostPosition'] = (out['Position_Change'] > 0).astype(int)
    out['NoPositionChange'] = (out['Position_Change'].abs() < eps).astype(int)

    # Phase/stint/pit-stop indicators.
    out['IsEarlyRace'] = (out['RaceProgress'] <= 0.25).astype(int)
    out['IsMidRace'] = ((out['RaceProgress'] > 0.25) & (out['RaceProgress'] < 0.75)).astype(int)
    out['IsLateRace'] = (out['RaceProgress'] >= 0.75).astype(int)
    out['IsFirstStint'] = (out['Stint'] == 1).astype(int)
    out['IsLaterStint'] = (out['Stint'] >= 3).astype(int)
    out['HasPitStop'] = (out['PitStop'] > 0).astype(int)
    out['PitStop_x_Stint'] = out['PitStop'] * out['Stint']
    out['Stint_x_TyreLife'] = out['Stint'] * out['TyreLife']

    # Tyre age buckets from the quantiles of the dataframe the features are built on.
    tyre_q25 = out['TyreLife'].quantile(0.25)
    tyre_q50 = out['TyreLife'].quantile(0.50)
    tyre_q75 = out['TyreLife'].quantile(0.75)
    out['IsFreshTyre'] = (out['TyreLife'] <= tyre_q25).astype(int)
    out['IsOldTyre'] = (out['TyreLife'] >= tyre_q50).astype(int)
    out['IsVeryOldTyre'] = (out['TyreLife'] >= tyre_q75).astype(int)
    out['LongNoPitStreak'] = (out['LapsSincePit'] >= tyre_q75).astype(int)

    # Compound flags and compound-age interactions.
    out['IsSoft'] = (out['Compound'] == 'SOFT').astype(int)
    out['IsMedium'] = (out['Compound'] == 'MEDIUM').astype(int)
    out['IsHard'] = (out['Compound'] == 'HARD').astype(int)
    out['Soft_x_TyreLife'] = out['IsSoft'] * out['TyreLife']
    out['Medium_x_TyreLife'] = out['IsMedium'] * out['TyreLife']
    out['Hard_x_TyreLife'] = out['IsHard'] * out['TyreLife']
    out['Soft_x_Progress'] = out['IsSoft'] * out['RaceProgress']
    out['Medium_x_Progress'] = out['IsMedium'] * out['RaceProgress']
    out['Hard_x_Progress'] = out['IsHard'] * out['RaceProgress']

    # Calendar/race type flags.
    out['IsRecentSeason'] = (out['Year'] >= 2024).astype(int)
    out['IsTesting'] = out['Race'].astype(str).str.contains('Testing', case=False, na=False).astype(int)
    return out


def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ]), categorical_cols),
        ],
        remainder='drop',
    )


def main():
    args = fold_args()
    train, test = read_train(), read_test()
    x = add_features(train.drop(columns=[TARGET, ID]))
    y = train[TARGET].astype(int).copy()
    x_test = add_features(test.drop(columns=[ID]))
    categorical_features = x.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = x.select_dtypes(include=[np.number]).columns.tolist()
    print(f'{x.shape[1]} features, categorical: {categorical_features}', flush=True)

    def fit_fold(fold, tr_idx, val_idx):
        model = Pipeline(steps=[
            ('preprocess', make_preprocessor(numeric_features, categorical_features)),
            ('model', XGBClassifier(**PARAMS)),
        ])
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        return model.predict_proba(x.iloc[val_idx])[:, 1], model.predict_proba(x_test)[:, 1]

    run_folds(FOLDER, FOLDER, fit_fold, args.folds, y=y.to_numpy(), clip=(1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
