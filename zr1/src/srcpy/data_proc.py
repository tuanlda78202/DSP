import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

# Load training transactions data
def load_training_data(file_name):
    return pd.read_csv(file_name)


# Load house properties data
def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, 
        dtype = {'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str})

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop


# Assign better names to all feature columns of 'properties' table
def rename_columns(prop):
    prop.rename(columns={
        'parcelid': 'parcel_id',                                    # Unique identifier of parcels
        'airconditioningtypeid': 'cooling_id',                      # Type of cooling system (if any), 1~13
        'architecturalstyletypeid': 'architecture_style_id',        # Architectural style of the home, 1~27
        'basementsqft': 'basement_sqft',                            # Size of the basement (1 square foot = 0.092903 m2)
        'bathroomcnt': 'bathroom_cnt',                              # Number of bathrooms (including fractional bathrooms)
        'bedroomcnt': 'bedroom_cnt',                                # Number of bedrooms
        'buildingclasstypeid': 'framing_id',                        # The building framing type, 1~5
        'buildingqualitytypeid': 'quality_id',                      # Building condition from best (lowest) to worst (highest)
        'calculatedbathnbr': 'bathroom_cnt_calc',                   # Same as "bathroom_cnt" ?
        'decktypeid': 'deck_id',                                    # Type of deck (if any)
        'finishedfloor1squarefeet': 'floor1_sqft',                  # Size of finished living area on first floor
        'calculatedfinishedsquarefeet': 'finished_area_sqft_calc',  # calculated total finished living area
        'finishedsquarefeet12': 'finished_area_sqft',               # Finished living area
        'finishedsquarefeet13': 'perimeter_area',                   # Perimeter living area
        'finishedsquarefeet15': 'total_area',                       # Total area
        'finishedsquarefeet50': 'floor1_sqft_unk',                  # Same as 'floor1_sqft' ?
        'finishedsquarefeet6': 'base_total_area',                   # Base unfinished and finished area
        'fips': 'fips',                                             # Federal Information Processing Standard code
        'fireplacecnt': 'fireplace_cnt',                            # Number of fireplaces in the home (if any)
        'fullbathcnt': 'bathroom_full_cnt',                         # Number of full bathrooms
        'garagecarcnt': 'garage_cnt',                               # Total number of garages
        'garagetotalsqft': 'garage_sqft',                           # Total size of the garages
        'hashottuborspa': 'spa_flag',                               # Whether the home has a hot tub or spa
        'heatingorsystemtypeid': 'heating_id',                      # Type of heating system, 1~25
        'latitude': 'latitude',                                     # Latitude of the middle of the parcel multiplied by 1e6
        'longitude': 'longitude',                                   # Longitude of the middle of the parcel multiplied by 1e6
        'lotsizesquarefeet': 'lot_sqft',                            # Area of the lot in sqft
        'poolcnt': 'pool_cnt',                                      # Number of pools in the lot (if any)
        'poolsizesum': 'pool_total_size',                           # Total size of the pools
        'pooltypeid10': 'pool_or_sht',                              # Spa or Hot Tub
        'pooltypeid2': 'pool_w_sht',                                # Pool with Spa/Hot Tub
        'pooltypeid7': 'pool_no_sht',                               # Pool without hot tub
        'propertycountylandusecode': 'country_landuse_code',        # County land use code i.e. it's zoning at the county level
        'propertylandusetypeid': 'landuse_type_id' ,                # Type of land use the property is zoned for, 25 categories
        'propertyzoningdesc': 'zoning_description',                 # Allowed land uses (zoning) for that property
        'rawcensustractandblock': 'census_raw',                     # Census tract and block ID combined - also contains block group assignment by extension
        'regionidcity': 'city_id',                                  # City in which the property is located (if any)
        'regionidcounty': 'country_id',                             # County in which the property is located
        'regionidneighborhood': 'neighborhood_id',                  # Neighborhood in which the property is located
        'regionidzip': 'region_zip',
        'roomcnt': 'room_cnt',                                      # Total number of rooms in the principal residence
        'storytypeid': 'story_id',                                  # Type of floors in a multi-story house, 1~35
        'threequarterbathnbr': 'bathroom_small_cnt',                # Number of 3/4 bathrooms
        'typeconstructiontypeid': 'construction_id',                # Type of construction material, 1~18
        'unitcnt': 'unit_cnt',                                      # Number of units the structure is built into (2=duplex, 3=triplex, etc)
        'yardbuildingsqft17': 'patio_sqft',                         # Patio in yard
        'yardbuildingsqft26': 'storage_sqft',                       # Storage shed/building in yard
        'yearbuilt': 'year_built',                                  # The year the principal residence was built
        'numberofstories': 'story_cnt',                             # Number of stories or levels the home has
        'fireplaceflag': 'fireplace_flag',                          # Whether the home has a fireplace
        'structuretaxvaluedollarcnt': 'tax_structure',              # The assessed value of the built structure on the parcel
        'taxvaluedollarcnt': 'tax_parcel',                          # The total tax assessed value of the parcel
        'assessmentyear': 'tax_year',                               # The year of the property tax assessment (2015 for 2016 data)
        'landtaxvaluedollarcnt': 'tax_land',                        # The assessed value of the land area of the parcel
        'taxamount': 'tax_property',                                # The total property tax assessed for that assessment year
        'taxdelinquencyflag': 'tax_overdue_flag',                   # Property taxes are past due as of 2015
        'taxdelinquencyyear': 'tax_overdue_year',                   # Year for which the unpaid properties taxes were due
        'censustractandblock': 'census_2'}, 
    inplace=True)


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


"""
    Compute and return datetime aggregate feature tables from a training set
    The returned tables can be joined for both training and inference
"""
def compute_datetime_aggregate_features(train):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(train.transaction_date).dt
    train['year'] = dt.year
    train['month'] = dt.month
    train['quarter'] = dt.quarter

    # Median log_error within the category
    logerror_year = train.groupby('year').log_error.median().to_frame() \
                                .rename(index=str, columns={"log_error": "log_error_year"})
    log_error_month = train.groupby('month').log_error.median().to_frame() \
                                .rename(index=str, columns={"log_error": "log_error_month"})
    log_error_quarter = train.groupby('quarter').logerror.median().to_frame() \
                                .rename(index=str, columns={"log_error": "log_error_quarter"})

    log_error_year.index = log_error_year.index.map(int)
    log_error_month.index = log_error_month.index.map(int)
    log_error_quarter.index = log_error_quarter.index.map(int)

    # Drop the temporary columns
    train.drop(['year', 'month', 'quarter'], axis=1, errors='ignore', inplace=True)

    return log_error_year, log_error_month, log_error_quarter


"""
    Add aggregate datetime features to a feature table
    The input table needs to have a 'transaction_date' columns
    The 'transaction_date' column is deleted from the table in the end
"""
def datetime_aggregate_features(df, log_error_year, log_error_month, log_error_quarter):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(df.transaction_date).dt
    df['year'] = dt.year
    df['month'] = dt.month
    df['quarter'] = dt.quarter

    # Join the aggregate features
    df = df.merge(how='left', right=log_error_year, on='year')
    df = df.merge(how='left', right=log_error_month, on='month')
    df = df.merge(how='left', right=log_error_quarter, on='quarter')

    # Drop the temporary columns
    df = df.drop(['year', 'month', 'quarter', 'transaction_date'], axis=1, errors='ignore')
    return df


# Add simple 'year', 'month', and 'quarter' categorical features to a DataFrame
def simple_datetime_features(df):
    dt = pd.to_datetime(df.transaction_date).dt
    df['year'] = (dt.year - 2016).astype(int)
    df['month'] = (dt.month).astype(int)
    df['quarter'] = (dt.quarter).astype(int)
    df.drop(['transaction_date'], axis=1, inplace=True)

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
