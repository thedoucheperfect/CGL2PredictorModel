import joblib
import json
import pandas as pd

class CGLPredictor:
    def __init__(self):
        # Load models
        self.rfr_main = joblib.load('Models/rfr_main.joblib')
        self.xgb_main = joblib.load('Models/xgb_main.joblib')
        self.scaler_main = joblib.load('Models/scaler.joblib')

        # Load firing models
        self.rfr_firing = joblib.load('Models/rfr_firing.joblib')
        self.xgb_firing = joblib.load('Models/xgb_firing.joblib')
        self.scaler_firing = joblib.load('Models/scaler_firing.joblib')

        # Load R² scores and maintain key order
        with open('Models/main_r2_scores.json') as f:
            self.main_scores = json.load(f)
            self.output_order = list(self.main_scores['RandomForest'].keys())

        with open('Models/firing_r2_scores.json') as f:
            self.firing_scores = json.load(f)

    def predict_main(self, input_data):
        # Scale input
        scaled_data = self.scaler_main.transform(pd.DataFrame([input_data]))
        
        # Get predictions and round to nearest integer
        rfr_pred = self.rfr_main.predict(scaled_data)[0].round().astype(int)
        xgb_pred = self.xgb_main.predict(scaled_data)[0].round().astype(int)

        # Create ordered predictions using stored key order
        final_pred = {}
        for output_name in self.output_order:
            idx = self.output_order.index(output_name)
            if self.main_scores['XGBoost'][output_name] > self.main_scores['RandomForest'][output_name]:
                final_pred[output_name] = xgb_pred[idx]
            else:
                final_pred[output_name] = rfr_pred[idx]
                
        return final_pred

    def predict_firing(self, nofs, speed):
        # Prepare and scale input
        firing_input = nofs + [speed]
        scaled_input = self.scaler_firing.transform([firing_input])

        # Select best model and round to nearest integer
        if self.firing_scores['XGBoost'] > self.firing_scores['RandomForest']:
            return round(self.xgb_firing.predict(scaled_input)[0])
        else:
            return round(self.rfr_firing.predict(scaled_input)[0])
