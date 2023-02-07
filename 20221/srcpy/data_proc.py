import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

# Load training transactions data
def load_training_data(file_name):
    return pd.read_csv(file_name).drop(['Unnamed: 0', "type", "rent_zestimate", "zestimate", "broker_name", "full_bathrooms"], axis=1)

def del_dup(df):
    l = list(df.loc[:, ["lat", "long"]].duplicated())
    store = []
    for index in range(len(l)):
        if l[index] == True:
            store.append(index)
    return df.drop(store)

# Convert TF to Float
def convert_true_to_float(df, col):
    df.loc[df[col] == 'True', col] = '1'
    df.loc[df[col] == 'False', col] = '0'
    df[col] = df[col].astype(float)
    return df

# Assign better names to all feature columns of 'properties' table
def rename_columns(prop):
    prop.rename(columns={
        'has_add_attributions': 'add_attr', 
        "latitude": "lat",
        "longitude": "long",
        "bathrooms": "bath", 
        "bedrooms": "bed",
        "living_area": "living",
        "lot_area": "lot_a",
        "num_fireplaces": "fireplace",
        "covered_spaces": "covered",
        "garage_spaces": "garage",
        "tax_assessed_value" : "tax_assessed",
        "lot_features": "lot_f",
        "architectural_style": "architectural",
        "year_built": "year",
        "sewer_info": "sewer",
        "water_info": "water",
        "appliances": "app",
        "interior_features": "interior"},
    inplace=True)

def encode_cat(df):
    df_cat = df.select_dtypes(exclude='number')
    catcol = list(df_cat.columns)
    for col in catcol:
        df[col] = df[col].astype('category').cat.codes
    return df

"""
    Convert some categorical variables to 'category' type
    Convert float64 variables to float32
    Note: In LightGBM, negative integer value for a categorical feature will be treated as missing value
"""
def retype_columns(prop):

    def norm_categorical(df, col):
        df[col] = df[col] - df[col].min()               # Convert the categories to have smaller labels (start from 0)
        df.loc[df[col].isnull(), col] = -1              # Fill missing data = -1
        df[col] = df[col].astype(int).astype('category')

    list_float2categorical = ['cooling_id', 'architecture_style_id', 'framing_id',
                             'heating_id', 'country_id', 'construction_id', 'fips', 
                             'landuse_type_id']

    # Convert categorical variables to 'category' type, and float64 variables to float32
    for col in prop.columns:
        if col in list_float2categorical:
            norm_categorical(prop, col)
        elif prop[col].dtype.name == 'float64':
            prop[col] = prop[col].astype(np.float32)

    gc.collect()


# Missing Data
def missingData(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    md = md[md["Percent"] > 0]
    plt.figure(figsize = (20, 8))
    plt.barh(md.index, md["Percent"],color="g",alpha=0.8)
    plt.title('Percent missing data by feature', fontsize=15)
    return md
