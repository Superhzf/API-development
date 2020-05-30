import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# import config
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, iqr, skew


def general_preprocess(df):
    # data cleaning
    df = df[df['CODE_GENDER'] != 'XNA']  # 4 people with XNA code gender
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]  # Max income in test is 4M
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Flag_document features - count and kurtosis
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    # Categorical age - based on target plot
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: _get_age_label(x, [27, 40, 50, 65, 99]))

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # Credit ratios
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Income ratios
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    # Time ratios
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    df['CREDIT_TO_ANNUITY_GROUP'] = df['CREDIT_TO_ANNUITY_RATIO'].apply(lambda x: _group_credit_to_annuity(x))
    return df



def chunk_groups(groupby_object, chunk_size):
    """Iterator that yields chunks of data with chunk_size."""
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_

def _get_age_label(days_birth, ranges):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    for label, max_age in enumerate(ranges):
        if age_years <= max_age:
            return label + 1
    else:
        return 0

def reduce_memory(df):
    """Reduce memory usage of a dataframe by setting data types. """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df
    
def _group_credit_to_annuity(x):
    """ Return the credit duration group label (int). """
    if x == np.nan: return 0
    elif x <= 6: return 1
    elif x <= 12: return 2
    elif x <= 18: return 3
    elif x <= 24: return 4
    elif x <= 30: return 5
    elif x <= 36: return 6
    else: return 7

class do_agg(BaseEstimator,TransformerMixin):
    def __init__(self,group_cols,counted,agg_name,mode):
        self.group_cols = group_cols
        self.counted = counted
        self.agg_name = agg_name
        self.mode = mode
        
    def fit(self,df,y=None):
        if self.mode == 'median':
            self.gp = df[self.group_cols + [self.counted]].groupby(self.group_cols)[self.counted].median().reset_index().rename(
                         columns={self.counted: self.agg_name})
        elif self.mode == 'std':
            self.gp = df[self.group_cols + [self.counted]].groupby(self.group_cols)[self.counted].std().reset_index().rename(
                          columns={self.counted: self.agg_name})
        else:
            raise Exception('mode is not understood')
            
    def transform(self,test_df):
        test_df = test_df.merge(self.gp, on=self.group_cols, how='left')
        return test_df
    
    def fit_transform(self,df,y=None):
        self.fit(df,y)
        return self.transform(df)

class do_label_encoder(BaseEstimator,TransformerMixin):
    def __init__(self,categorical_columns=None):
        self.categorical_columns = categorical_columns
        self.label_encoders = {}
        
    def fit(self,df,y=None):
        if not self.categorical_columns:
            self.categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            self.label_encoders[col]=le_dict
            
    def transform(self,test_df):
        for col in self.categorical_columns:
            test_df[col] = test_df[col].apply(lambda x: self.label_encoders[col].get(x, -1))
#             test_df[col]=self.label_encoders[col].transform(test_df[col].astype(str))
        return test_df
   
    def fit_transform(self,df,y=None):
        self.fit(df,y)
        return transform(df)
            
            
def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize.
    Arguments:
        df: DataFrame.
        categorical_columns: List of column names;
        if None all columns with object datatype will be considered.
    Returns:
        df: Same DataFrame with the encoded features
        categorical_columns: List with the names of OHE columns.
    """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], _ = pd.factorize(df[col])
    return df, categorical_columns

def _drop_application_columns(df):
    """ Drop a few noise features. """
    drop_list = [
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE',
        'LIVE_REGION_NOT_WORK_REGION', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK',
        'COMMONAREA_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
        'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE', 'ELEVATORS_MEDI', 'EMERGENCYSTATE_MODE',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
    ]
    # Drop most flag document columns
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns.
    Arguments:
        df: DataFrame.
        categorical_columns: List of column names;
        if None all columns with object datatype will be considered.
        nan_as_category: If True add column for missing values (boolean)
    Returns:
        df: Same DataFrame with the encoded features
        categorical_columns: List with the names of OHE columns.
    """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """Group DataFrame and perform aggregations.
    Arguments:
        df_to_agg: DataFrame to be grouped.
        prefix: New features name prefix
        aggregations: Dictionary or list of aggregations (see pandas aggregate)
        aggregate_by: Column to group DataFrame
    Returns:
        agg_df: DataFrame with new features
    """
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """Group DataFrame, perform aggregations and merge with the second one.
    Arguments:
        df_to_agg: DataFrame to be grouped.
        df_to_merge: DataFrame where agg will be merged.
        prefix: Prefix for new features names (string)
        aggregations: Dictionary or list of aggregations (see pandas aggregate)
        aggregate_by: Column name to group DataFrame  (string)
    Returns:
        df_to_merge: Second dataframe with the aggregated features from the first.
    """
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)

def do_sum(df, group_cols, counted, agg_name):
    """Add the sum for each group for a given feature in a DataFrame.
    Arguments:
        df: DataFrame to group and add the sum feature.
        group_cols: List with column or columns names to group by.
        counted: Feature name to get the sum (string).
        agg_name: New feature name (string)
    Returns:
        df: Same DataFrame with the new feature
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    """Apply the given function using multiprocessing (in parallel).
    Arguments:
        groups: pandas Groupby object.
        func: Function to apply.
        index_name: pandas Index (string).
        num_workers: Number of jobs (threads). If zero, config value will be used.
    Returns:
        features: Same DataFrame from arguments with new features.
    """
#     if num_workers <= 0: num_workers = config.NUM_THREADS
    if num_workers <= 0: num_workers = 4
    #n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def add_trend_feature(features, gr, feature_name, prefix):
    """Return linear regression parameters (linear trend) for given feature.
    Arguments:
        features: DataFrame where trends will be add.
        gr: pandas Groupby object.
        feature_name: Feature for calculating trends (string).
        prefix: Prefix for the name of the new features (string).
    Returns:
        features: Same DataFrame from arguments with new features.
    """
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    """Generate all aggregation features in aggs list for the given feature.
    Arguments:
        features: DataFrame where features will be add.
        gr_: pandas Groupby object.
        feature_name: Feature used for aggregation (string).
        aggs: List of strings with pandas aggregations names.
        prefix: Prefix for the name of the new features (string).
    Returns:
        features: Same DataFrame from arguments with new features.
    """
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features
