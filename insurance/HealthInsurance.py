import pickle
import pandas as pd
import numpy  as np

class HealthInsurance:
    def __init__(self):
        self.home_path = 'parameter/'
        self.annual_premium_scaler = pickle.load(open(self.home_path + 'annual_premium_scaler.pkl', 'rb'))
        self.age_scaler            = pickle.load(open(self.home_path + 'age_scaler.pkl', 'rb'))
        self.vintage_scaler        = pickle.load(open(self.home_path + 'vintage_scaler.pkl', 'rb'))
        self.gender_encoder        = pickle.load(open(self.home_path + 'gender_encoder.pkl', 'rb'))
        self.region_code_encoder   = pickle.load(open(self.home_path + 'region_code_encoder.pkl', 'rb'))
        self.policy_sales_encoder  = pickle.load(open(self.home_path + 'policy_sales_encoder.pkl', 'rb'))
        
        
    def data_cleaning(self, data):
        # columns in snakecase
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code',
                    'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
                    'policy_sales_channel', 'vintage']

        data.columns = cols_new
        
        return data
        

    def feature_engineering(self, data):
        # vehicle age
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else
                                                      'between_1_2_years' if x == '1-2 Year' else 'below_1_year')

        # vehicle damage
        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        return data
    
    
    def data_preparation(self, data):
        # annual_premium
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)

        # age (since its distribution is not a normal one)
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)

        # gender
        data['gender'] = self.gender_encoder.transform(data['gender'])

        # region_code - Target Enconder
        data['region_code'] = data['region_code'].astype(str)
        data['region_code'] = self.region_code_encoder.transform(data['region_code'])

        # vehicle_age - Ordinal Encoding
        vehicle_age_dict = {'below_1_year': 1, 'between_1_2_years': 2, 'over_2_years': 3}
        data['vehicle_age'] = data['vehicle_age'].map(vehicle_age_dict)

        # policy_sales_channel - Frequency Encoding
        data['policy_sales_channel'] = data['policy_sales_channel'].astype(str)
        data = self.policy_sales_encoder.transform(data)
        data = data.replace(np.nan, 0)
        
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage',
                         'policy_sales_channel', 'previously_insured']
        
        return data[cols_selected]
    
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict_proba(test_data)
        
        # join score into the original data
        original_data['score'] = pred[:,1].tolist()

        return original_data.to_json(orient='records', date_format='iso')
