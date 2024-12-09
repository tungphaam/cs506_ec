import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from category_encoders import TargetEncoder

def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess the training and test data with advanced features."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, test_df]:
        # Temporal features
        df['trans_date_time'] = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'])
        df['hour'] = df['trans_date_time'].dt.hour
        df['day_of_week'] = df['trans_date_time'].dt.dayofweek
        df['month'] = df['trans_date_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Enhanced temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['is_rush_hour'] = df['hour'].isin([8, 9, 17, 18]).astype(int)
        df['is_lunch_time'] = df['hour'].isin([11, 12, 13]).astype(int)
        df['is_evening'] = df['hour'].isin([19, 20, 21]).astype(int)
        df['is_early_morning'] = df['hour'].isin([6, 7]).astype(int)
        
        # Improved date-based features
        df['day_of_month'] = df['trans_date_time'].dt.day
        df['week_of_year'] = df['trans_date_time'].dt.isocalendar().week
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Enhanced age-related features
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (pd.Timestamp('now') - df['dob']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        df['age_bin'] = pd.qcut(df['age'], q=15, labels=False, duplicates='drop')
        df['is_senior'] = (df['age'] >= 65).astype(int)
        df['is_young_adult'] = ((df['age'] >= 18) & (df['age'] <= 25)).astype(int)
        
        # Improved location features
        df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        df['distance_bin'] = pd.qcut(df['distance'], q=15, labels=False, duplicates='drop')
        df['pop_density'] = df['city_pop'] / (df['distance'] + 1)
        df['high_density_area'] = (df['pop_density'] > df['pop_density'].median()).astype(int)
        
        # Enhanced amount features
        df['amount_bin'] = pd.qcut(df['amt'], q=25, labels=False, duplicates='drop')
        df['amount_log'] = np.log1p(df['amt'])
        df['is_large_transaction'] = (df['amt'] > df['amt'].quantile(0.95)).astype(int)
        
        # More sophisticated interaction features
        df['amount_per_pop'] = df['amt'] / (df['city_pop'] + 1)
        df['distance_amount_ratio'] = df['distance'] / (df['amt'] + 1)
        df['amount_per_age'] = df['amt'] / (df['age'] + 1)
        df['pop_amount_ratio'] = df['city_pop'] / (df['amt'] + 1)
        df['high_risk_combo'] = ((df['is_night'] == 1) & (df['is_large_transaction'] == 1)).astype(int)
        
        # Location risk features
        df['zip_length'] = df['zip'].astype(str).str.len()
        df['state_length'] = df['state'].astype(str).str.len()

    # Enhanced target encoding
    categorical_columns = ['category', 'state', 'job', 'merchant']
    target_encoder = TargetEncoder(smoothing=10)
    
    if 'is_fraud' in train_df.columns:
        train_encoded = target_encoder.fit_transform(train_df[categorical_columns], train_df['is_fraud'])
        test_encoded = target_encoder.transform(test_df[categorical_columns])
    else:
        train_encoded = target_encoder.fit_transform(train_df[categorical_columns])
        test_encoded = target_encoder.transform(test_df[categorical_columns])
    
    for col in categorical_columns:
        train_df[f'{col}_encoded'] = train_encoded[col]
        test_df[f'{col}_encoded'] = test_encoded[col]

    feature_columns = [
        'amt', 'hour', 'day_of_week', 'month', 'age', 'distance',
        'city_pop', 'is_weekend', 'is_night', 'hour_sin', 'hour_cos',
        'is_rush_hour', 'is_lunch_time', 'is_evening', 'is_early_morning',
        'distance_bin', 'amount_bin', 'pop_density', 'amount_per_pop',
        'distance_amount_ratio', 'amount_log', 'amount_per_age',
        'pop_amount_ratio', 'age_bin', 'day_of_month', 'week_of_year',
        'is_month_start', 'is_month_end', 'is_senior', 'is_young_adult',
        'high_density_area', 'is_large_transaction', 'high_risk_combo'
    ] + [f'{col}_encoded' for col in categorical_columns]

    # Improved scaling with PowerTransformer
    power_transformer = PowerTransformer(method='yeo-johnson')
    numerical_columns = [col for col in feature_columns if '_bin' not in col and '_encoded' not in col 
                        and not col.startswith('is_')]
    
    train_df[numerical_columns] = power_transformer.fit_transform(train_df[numerical_columns])
    test_df[numerical_columns] = power_transformer.transform(test_df[numerical_columns])
    
    X_train = train_df[feature_columns]
    y_train = train_df['is_fraud'] if 'is_fraud' in train_df.columns else None
    X_test = test_df[feature_columns]
    
    return X_train, y_train, X_test, test_df['id']

def create_ensemble_model(X_train, y_train):
    """Create an optimized ensemble of XGBoost and LightGBM models."""
    # Enhanced XGBoost parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,          # Increased from 400
        max_depth=11,               # Increased from 7
        learning_rate=0.04,        # Decreased from 0.05 for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=7,        # Increased from 3
        scale_pos_weight=21,       # Increased from 12
        gamma=0.3,                 # Increased from 0.2
        reg_alpha=0.2,             # Increased from 0.1
        reg_lambda=1.2,            # Increased from 1.0
        random_state=42,
        tree_method='hist',        # Added for faster training
        max_bin=256               # Added for better splits
    )
    
    # Enhanced LightGBM parameters
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,          # Increased from 400
        max_depth=11,               # Increased from 7
        learning_rate=0.04,        # Decreased from 0.05
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=7,        # Increased from 3
        scale_pos_weight=21,       # Increased from 12
        random_state=42,
        num_leaves=35,            # Increased from 31
        reg_alpha=0.2,            # Increased from 0.1
        reg_lambda=1.2,           # Increased from 1.0
        boosting_type='gbdt',
        objective='binary',
        metric='binary_logloss',
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_jobs=-1
    )
    
    # Adjusted ensemble weights based on individual model performance
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        weights=[0.6, 0.4]        # Adjusted from [0.55, 0.45]
    )
    
    # Enhanced SMOTE parameters
    smote = SMOTE(
        random_state=48,
        k_neighbors=11,            # Increased from 5
        sampling_strategy=0.6,    # Increased from 0.4
        n_jobs=-1
    )
    
    # Apply SMOTE and fit the ensemble
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("Fitting ensemble model...")
    ensemble.fit(X_resampled, y_resampled)
    
    return ensemble

def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    submission_path = 'submission.csv'
    
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_ids = load_and_preprocess_data(train_path, test_path)
    
    print("Training ensemble model...")
    model = create_ensemble_model(X_train, y_train)
    
    print("Generating submission file...")
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        'id': test_ids,
        'is_fraud': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

if __name__ == "__main__":
    main()