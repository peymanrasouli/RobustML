import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification, make_circles

## Preparing Adult dataset
def PrepareAdult(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values=' ?')

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)
    cat_to_num = {
                  "sex": {" Male": 0,
                          " Female": 1}
                 }
    df = df.replace(cat_to_num)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age', 'fnlwgt', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset

## Preparing Default of Credit Card Clients dataset
def PrepareCreditCardDefault(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=',')

    df.rename(columns={ 'default payment next month': 'class'}, inplace=True)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['LIMIT_BAL', 'SEX', 'EDUCATION','MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4',
                           'PAY_5', 'PAY_6' , 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                           'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset


## Preparing Heart Disease dataset
def PrepareHeartDisease(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values='?')
    df.rename(columns={'num': 'class'}, inplace=True)

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df.loc[df[class_name] != 0, class_name] = 1
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age','sex', 'trestbps','chol', 'fbs', 'thalach', 'exang', 'oldpeak','slope', 'ca']
    discrete_features = []
    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }


    return dataset

## Preparing COMPAS dataset
def PrepareCOMPAS(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', skipinitialspace=True)

    ## Data cleaning
    # handling missing values and converting feature types
    columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest',
               'c_jail_in', 'c_jail_out', 'c_charge_degree', 'is_recid', 'is_violent_recid',
               'two_year_recid', 'decile_score', 'score_text']
    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
    cat_to_num = {
                  "sex": {"Male": 0,
                          "Female": 1}
                  }

    df = df.replace(cat_to_num)

    # classifying instances with respect to recidivism score
    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'

    df['class'] = df['decile_score'].apply(get_class)

    # removing useless columns
    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['sex', 'age', 'length_of_stay', 'days_b_screening_arrest', 'is_recid',  'is_violent_recid',
                           'priors_count', 'two_year_recid']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset

## Preparing German Credit dataset
def PrepareGermanCredit(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',')
    df.rename(columns={'default': 'class'}, inplace=True)

    cat_to_num = {
                  "account_check_status": {"no checking account": 0,
                                           "< 0 DM": 1,
                                           "0 <= ... < 200 DM": 2,
                                           ">= 200 DM / salary assignments for at least 1 year":3},

                  "savings": {"unknown/ no savings account": 0,
                              "... < 100 DM": 1,
                              "100 <= ... < 500 DM": 2,
                              "500 <= ... < 1000 DM ": 3,
                              ".. >= 1000 DM ": 4},

                  "present_emp_since": {"unemployed": 0,
                                        "... < 1 year ": 1,
                                        "1 <= ... < 4 years": 2,
                                        "4 <= ... < 7 years": 3,
                                        ".. >= 7 years": 4},
                  "telephone": {"none": 0,
                                "yes, registered under the customers name ": 1},

                  "foreign_worker": {"no": 0,
                                     "yes": 1}

                  }

    df = df.replace(cat_to_num)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['account_check_status', 'duration_in_month', 'credit_amount', 'savings', 'present_emp_since',
                           'installment_as_income_perc', 'present_res_since', 'age', 'credits_this_bank',
                           'people_under_maintenance', 'telephone','foreign_worker']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset


## Preparing circles dataset
def PrepareCircles(dataset_path, dataset_name):

    ## Creating circles dataset
    data = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
    df = pd.DataFrame(data=np.c_[data[0],data[1]], columns=['x1', 'x2','class'])

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['x1', 'x2']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': 'circles',
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset

## Preparing linearly separable dataset
def PrepareLinearlySeparable(dataset_path, dataset_name):

    ## Creating circles dataset
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    df = pd.DataFrame(data=np.c_[X,y], columns=['x1', 'x2','class'])

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['x1', 'x2']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =MinMaxScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    ## Returning dataset information
    dataset = {
        'name': 'linearly-separable',
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset