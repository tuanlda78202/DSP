import numpy as np 
import pandas as pd 
import gc

# Rename & retype the feature columns; also unify representations of missing values
def get_landuse_code_df(prop_2016, prop_2017):

    # 2016
    temp = prop_2016.groupby('country_landuse_code')['country_landuse_code'].count()
    landuse_codes = list(temp[temp >= 300].index)

    # 2017
    temp = prop_2017.groupby('country_landuse_code')['country_landuse_code'].count()
    landuse_codes += list(temp[temp >= 300].index)
    landuse_codes = list(set(landuse_codes))

    # Country land use DF 
    df_landuse_codes = pd.DataFrame({'country_landuse_code': landuse_codes,
                                     'country_landuse_code_id': range(len(landuse_codes))})
    return df_landuse_codes

# Get zoning desc code 
def get_zoning_desc_code_df(prop_2016, prop_2017):

    # 2016
    temp = prop_2016.groupby('zoning_description')['zoning_description'].count()
    zoning_codes = list(temp[temp >= 5000].index)

    # 2017
    temp = prop_2017.groupby('zoning_description')['zoning_description'].count()
    zoning_codes += list(temp[temp >= 5000].index)
    zoning_codes = list(set(zoning_codes))

    # Country land use DF 
    df_zoning_codes = pd.DataFrame({'zoning_description': zoning_codes,
                                     'zoning_description_id': range(len(zoning_codes))})
    return df_zoning_codes

# Process columns 
def process_columns(df, df_landuse_codes, df_zoning_codes):
    df = df.merge(how='left', right=df_landuse_codes, on='country_landuse_code')
    df = df.drop(['country_landuse_code'], axis=1)
    
    df = df.merge(how='left', right=df_zoning_codes, on='zoning_description')
    df = df.drop(['zoning_description'], axis=1)
    
    df.loc[df.country_id == 3101, 'country_id'] = 0
    df.loc[df.country_id == 1286, 'country_id'] = 1
    df.loc[df.country_id == 2061, 'country_id'] = 2
    
    df.loc[df.landuse_type_id == 279, 'landuse_type_id'] = 261
    return df


# Feature engineering 
def feature_engineering(prop_2016, prop_2017):
    for prop in [prop_2016, prop_2017]:
        # Avg garage size 
        prop['avg_garage_size'] = prop['garage_sqft'] / prop['garage_cnt']
    
        # Property tax per sqft
        prop['property_tax_per_sqft'] = prop['tax_property'] / prop['finished_area_sqft_calc']
    
        # Rotated Coordinates
        prop['location_sum'] = prop['latitude'] + prop['longitude']
        prop['location_minus'] = prop['latitude'] - prop['longitude']
        prop['location_sum05'] = prop['latitude'] + 0.5 * prop['longitude']
        prop['location_minus05'] = prop['latitude'] - 0.5 * prop['longitude']
    
        # 'finished_area_sqft' and 'total_area' cover only a strict subset of 'finished_area_sqft_calc' in terms of non-missing values. 
        # Also, when both fields are not null, the values are always the same.
        # So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
        # If there're some patterns in when the values are missing, we can add two isMissing binary features
        prop['missing_finished_area'] = prop['finished_area_sqft'].isnull().astype(np.float32)
        prop['missing_total_area'] = prop['total_area'].isnull().astype(np.float32)
        prop.drop(['finished_area_sqft', 'total_area'], axis=1, inplace=True)
    
        # Same as above, 'bathroom_cnt' covers everything that 'bathroom_cnt_calc' has
        # So we can safely drop 'bathroom_cnt_calc' and optionally add an isMissing feature
        prop['missing_bathroom_cnt_calc'] = prop['bathroom_cnt_calc'].isnull().astype(np.float32)
        prop.drop(['bathroom_cnt_calc'], axis=1, inplace=True)
    
        # 'room_cnt' has many zero or missing values
        # On the other hand, 'bathroom_cnt' and 'bedroom_cnt' have few zero or missing values
        # Add an derived room_cnt feature by adding bathroom_cnt and bedroom_cnt
        prop['derived_room_cnt'] = prop['bedroom_cnt'] + prop['bathroom_cnt']
    
        # Average area in sqft per room
        mask = (prop.room_cnt >= 1)  # avoid dividing by zero
        prop.loc[mask, 'avg_area_per_room'] = prop.loc[mask, 'finished_area_sqft_calc'] / prop.loc[mask, 'room_cnt']
    
        # Use the derived room_cnt to calculate the avg area again
        mask = (prop.derived_room_cnt >= 1)
        prop.loc[mask,'derived_avg_area_per_room'] = prop.loc[mask,'finished_area_sqft_calc'] / prop.loc[mask,'derived_room_cnt']
    return prop_2016, prop_2017


# Compute region-based aggregate features
def region_aggregate_features(df, group_col, agg_cols):
    df[group_col + '-groupcnt'] = df[group_col].map(df[group_col].value_counts())
    new_columns = []  # New feature columns added to the DataFrame

    for col in agg_cols:
        aggregates = df.groupby(group_col, as_index=False)[col].agg([np.mean])
        aggregates.columns = [group_col + '-' + col + '-' + s for s in ['mean']]
        new_columns += list(aggregates.columns)
        df = df.merge(how='left', right=aggregates, on=group_col)
        
    for col in agg_cols:
        mean = df[group_col + '-' + col + '-mean']
        diff = df[col] - mean
        
        df[group_col + '-' + col + '-' + 'diff'] = diff
        if col != 'year_built':
            df[group_col + '-' + col + '-' + 'percent'] = diff / mean
        
    # Set the values of the new features to NaN if the groupcnt is too small (prevent overfitting)
    threshold = 100
    df.loc[df[group_col + '-groupcnt'] < threshold, new_columns] = np.nan
    
    # Drop the mean features since they turn out to be not useful
    df.drop([group_col+'-'+col+'-mean' for col in agg_cols], axis=1, inplace=True)
    
    gc.collect()
    return df