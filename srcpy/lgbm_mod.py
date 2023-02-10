import pandas as pd
import numpy as np

# Save LightGBM models to files
def save_models(models):
    for i, model in enumerate(models):
        model.save_model('checkpoints/lgb_' + str(i))
    print("Saved {} LightGBM models to files.".format(len(models)))

# Load LightGBM models from files
def load_models(paths):
    models = []
    for path in paths:
        model = lgb.Booster(model_file=path)
        models.append(model)
    return models

# Drop id and label columns + Feature selection for LightGBM
def lgb_drop_features(features):
    # id and label (not features)
    unused_feature_list = ['parcel_id', 'log_error']

    # too many missing (LightGBM is robust against bad/unrelated features, so this step might not be needed)
    missing_list = ['framing_id', 'architecture_style_id', 'story_id', 'perimeter_area', 'basement_sqft', 'storage_sqft']
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = ['fireplace_flag', 'deck_id', 'pool_unk_1', 'construction_id', 'county_id', 'fips']
    unused_feature_list += bad_feature_list

    # really hurts performance
    unused_feature_list += ['county_landuse_code_id', 'zoning_description_id']

    return features.drop(unused_feature_list, axis=1, errors='ignore')


# Helper method that prepares 2016 and 2017 properties features for inference
def transform_test_features(features_2016, features_2017):
    test_features_2016 = lgb_drop_features(features_2016)
    test_features_2017 = lgb_drop_features(features_2017)
    
    test_features_2016['year'] = 0
    test_features_2017['year'] = 1
    
    # 11 & 12 lead to unstable results, probably due to the fact that there are few training examples for them
    test_features_2016['month'] = 10
    test_features_2017['month'] = 10
    
    test_features_2016['quarter'] = 4
    test_features_2017['quarter'] = 4
    
    return test_features_2016, test_features_2017

"""
    Helper method that makes predictions on the test set and exports results to csv file
    'models' is a list of models for ensemble prediction (len=1 means using just a single model)
"""
def predict_and_export(models, features_2016, features_2017, file_name):
    # Construct DataFrame for prediction results
    submission_2016 = pd.DataFrame()
    submission_2017 = pd.DataFrame()
    submission_2016['ParcelId'] = features_2016.parcel_id
    submission_2017['ParcelId'] = features_2017.parcel_id
    
    test_features_2016, test_features_2017 = transform_test_features(features_2016, features_2017)
    
    pred_2016, pred_2017 = [], []
    for i, model in enumerate(models):
        print("Start model {} (2016)".format(i))
        pred_2016.append(model.predict(test_features_2016, predict_disable_shape_check= True))
        print("Start model {} (2017)".format(i))
        pred_2017.append(model.predict(test_features_2017, predict_disable_shape_check= True))
    
    # Take average across all models
    mean_pred_2016 = np.mean(pred_2016, axis=0)
    mean_pred_2017 = np.mean(pred_2017, axis=0)
    
    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]
    submission_2016['201611'] = submission_2016['201610']
    submission_2016['201612'] = submission_2016['201610']

    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]
    submission_2017['201711'] = submission_2017['201710']
    submission_2017['201712'] = submission_2017['201710']
    
    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')
    
    print("Length of submission DataFrame: {}".format(len(submission)))
    print("Submission header:")
    print(submission.head())
    submission.to_csv(file_name, index=False)
    return submission, pred_2016, pred_2017  # Return the results so that we can analyze or sanity check it