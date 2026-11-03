# The Preprocessing class of the public notebook "Smartphone Addiction | XGB" (version 3), cell 10, unchanged:
# https://www.kaggle.com/code/mikhailnaumov/smartphone-addiction-xgb
import numpy as np
import pandas as pd


class Preprocessing:
    def __init__(self, target):
        self.target = target
        self._fitted = False

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series, orig: pd.DataFrame = None):
        self.fit(X_train, y_train, orig=orig)
        return self.transform(X_train)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, orig: pd.DataFrame = None):
        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()

        self.num_features_ = self.X_train_.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features_ = self.X_train_.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        
        self._orig_stats = {}

        if orig is not None:
            self.orig_ = orig.copy()
            self._fit_orig_target_stats(self.cat_features_ + self.num_features_)

        self._fit_feature_engineering(self.X_train_)
        X_fe_train = self._apply_feature_engineering(self.X_train_)

        self._fit_frequency_encoding(X_fe_train)
        X_fe_train = self._apply_frequency_encoding(X_fe_train)
        
        X_fe_train = self._finalize_types(X_fe_train, fit=True)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Preprocessing is not fitted. Call fit() first.")

        X = X.copy()

        X = self._apply_orig_target_stats(X)

        X = self._apply_feature_engineering(X)

        X = self._apply_frequency_encoding(X)

        X = self._finalize_types(X, fit=False)

        for c in self.cat_features:
            X[c] = X[c].astype(object).fillna('NaN').astype('category')

        return X, self.cat_features, self.num_features

    def _fit_orig_target_stats(self, cols):
        self._orig_global_mean = float(self.orig_[self.target].mean())
        for c in cols:
            if c in self.orig_.columns:
                self._orig_stats[c] = self.orig_.groupby(c, observed=False)[self.target].mean()
            else:
                self._orig_stats[c] = pd.Series(dtype=float)

    def _apply_orig_target_stats(self, df):
        df = df.copy()
        for c, ser in self._orig_stats.items():
            col = f"{c}_org_mean"
            if c in df.columns and not ser.empty:
                df[col] = df[c].map(ser).astype(float).fillna(self._orig_global_mean)
            else:
                df[col] = self._orig_global_mean
        return df

    def _fit_feature_engineering(self, X_train):
        X_train = X_train.copy()

        self._highcard_num = [c for c in self.num_features_ if X_train[c].nunique(dropna=False) > 20]
        self.category_map = {}

    def _apply_feature_engineering(self, df):
        df = df.copy()

        self.num_to_cat = []
        for col in self.num_features_:
            cat_name = f"{col}_cat"
            df[cat_name] = df[col].fillna('NaN').astype(str)
            self.num_to_cat.append(cat_name)

        for c in self.num_features_:
            for i in [0,1]:
                df[f"{c}_round{i}"] = df[c].round(i)

        social = df['social_media_hours']
        gaming = df['gaming_hours']
        work = df['work_study_hours']
        daily = df['daily_screen_time_hours']
        sleep = df['sleep_hours']
        weekend = df['weekend_screen_time']
        opens = df['app_opens_per_day']
        notifications = df['notifications_per_day']
    
        df['total_breakdown_hours'] = social.fillna(0) + gaming.fillna(0) + work.fillna(0)
        df['social_ratio'] = social / (daily + 1e-5)
        df['gaming_ratio'] = gaming / (daily + 1e-5)
        df['work_ratio'] = work / (daily + 1e-5)
        df['unaccounted_screen_time'] = daily - df['total_breakdown_hours']
        df['screen_to_sleep_ratio'] = daily / (sleep + 1e-5)
        df['weekend_vs_daily_ratio'] = weekend / (daily + 1e-5)
        df['app_opens_per_hour'] = opens / (daily + 1e-5)
        df['notifications_per_hour'] = notifications / (daily + 1e-5)
    
        COMP = ['social_media_hours', 'gaming_hours', 'work_study_hours']
        n_missing_components = df[COMP].isna().sum(axis=1)
        missing_daily = daily.isna()
        known_component_sum = df[COMP].sum(axis=1, min_count=1)
        slack = daily - known_component_sum
        daily_missing_with_components = missing_daily & (n_missing_components == 0)
        one_component_missing = (~missing_daily) & (n_missing_components == 1)
        all_screen_known = (~missing_daily) & (n_missing_components == 0)
    
        df['ci_daily_lb'] = np.where(daily_missing_with_components, known_component_sum, np.nan)
        df['ci_missing_comp_ub'] = np.where(one_component_missing, slack, np.nan)
        df['ci_missing_comp_mid'] = np.where(one_component_missing, np.clip(slack / 2, 0, None), np.nan)
        df['ci_interval_width'] = np.where(one_component_missing, slack, np.nan)
        df['ci_slack_exact'] = np.where(all_screen_known, slack, np.nan)
        df['ci_case'] = np.select([daily_missing_with_components, one_component_missing], [1.0, 2.0], default=0.0).astype('float32')
        df['ci_n_missing'] = (missing_daily.astype('int8') + n_missing_components).astype('float32')
        
        df['missing_count'] = df[self.num_features_+self.cat_features_].isna().sum(axis=1).astype('int8')
            
        drop = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
        if drop:
            df.drop(drop, axis=1, inplace=True)

        return df

    def _fit_frequency_encoding(self, X):
        all_cats = list(dict.fromkeys(self.cat_features_+self.num_to_cat))
        self._fe_cols = [c for c in all_cats if c in X.columns]

        self._freq_encodings = {}
        for c in self._fe_cols:
            self._freq_encodings[c] = X[c].value_counts(normalize=True).to_dict()

    def _apply_frequency_encoding(self, X):
        X = X.copy()
        for c in getattr(self, "_fe_cols", []):
            mapping = self._freq_encodings.get(c, {})
            if c in X.columns:
                X[f"{c}_fe"] = X[c].map(mapping).astype(float).fillna(0.0)
            else:
                X[f"{c}_fe"] = 0.0
        return X
        
    def _finalize_types(self, X, fit=False):
        self.num_features = X.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        return X
