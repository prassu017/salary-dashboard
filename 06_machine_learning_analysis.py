"""
BUS AN 512 Data Management and SQL - Group Project
Data Science Salary Analysis (2020-2025)
Machine Learning Analysis Script

This script provides:
- Salary prediction models
- Feature importance analysis
- Model performance evaluation
- Advanced ML insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SalaryMLAnalyzer:
    def __init__(self, file_path='salaries.csv'):
        """Initialize the ML analyzer"""
        self.df = pd.read_csv(file_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for machine learning"""
        print("Preparing data for machine learning...")
        
        # Clean data
        self.df = self.df.dropna(subset=['salary_in_usd', 'work_year', 'experience_level'])
        
        # Create features
        self.df['work_year'] = pd.to_datetime(self.df['work_year'], format='%Y')
        self.df['year'] = self.df['work_year'].dt.year
        
        # Encode categorical variables
        self.label_encoders = {}
        categorical_cols = ['experience_level', 'employment_type', 'company_size', 'employee_residence']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Create remote work categories
        self.df['remote_category'] = self.df['remote_ratio'].map({
            0: 'On-site',
            50: 'Hybrid',
            100: 'Fully Remote'
        })
        le_remote = LabelEncoder()
        self.df['remote_category_encoded'] = le_remote.fit_transform(self.df['remote_category'])
        self.label_encoders['remote_category'] = le_remote
        
        # Select features for ML
        self.feature_cols = [
            'year', 'experience_level_encoded', 'employment_type_encoded',
            'remote_ratio', 'company_size_encoded', 'employee_residence_encoded',
            'remote_category_encoded'
        ]
        
        self.X = self.df[self.feature_cols]
        self.y = self.df['salary_in_usd']
        
        print(f"Features: {self.feature_cols}")
        print(f"Data shape: {self.X.shape}")
        
    def train_models(self):
        """Train multiple ML models for salary prediction"""
        print("\n=== TRAINING MACHINE LEARNING MODELS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: ${rmse:,.0f}")
            print(f"MAE: ${mae:,.0f}")
            print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        
        return results
    
    def model_comparison(self):
        """Compare model performance"""
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison dataframe
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'CV R² Mean': metrics['cv_mean'],
                'CV R² Std': metrics['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Visualize model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² scores
        axes[0,0].bar(comparison_df['Model'], comparison_df['R² Score'])
        axes[0,0].set_title('R² Scores by Model')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # RMSE
        axes[0,1].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0,1].set_title('RMSE by Model')
        axes[0,1].set_ylabel('RMSE (USD)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[1,0].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[1,0].set_title('MAE by Model')
        axes[1,0].set_ylabel('MAE (USD)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # CV R²
        axes[1,1].bar(comparison_df['Model'], comparison_df['CV R² Mean'])
        axes[1,1].set_title('Cross-Validation R² by Model')
        axes[1,1].set_ylabel('CV R² Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get best model (Random Forest or XGBoost)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        
        print(f"Analyzing feature importance for {best_model_name}")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            # For linear models, use absolute coefficients
            importance = np.abs(best_model.coef_)
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance:")
        print(feature_importance_df.to_string(index=False))
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def salary_prediction_insights(self):
        """Generate insights from salary predictions"""
        print("\n=== SALARY PREDICTION INSIGHTS ===")
        
        # Use best model for predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        y_pred = self.results[best_model_name]['predictions']
        
        # Analyze prediction errors
        errors = self.y_test - y_pred
        error_stats = {
            'Mean Error': errors.mean(),
            'Std Error': errors.std(),
            'Min Error': errors.min(),
            'Max Error': errors.max(),
            'Mean Absolute Error': np.abs(errors).mean()
        }
        
        print("Prediction Error Analysis:")
        for stat, value in error_stats.items():
            print(f"{stat}: ${value:,.0f}")
        
        # Create error analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0,0].scatter(self.y_test, y_pred, alpha=0.5)
        axes[0,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Salary')
        axes[0,0].set_ylabel('Predicted Salary')
        axes[0,0].set_title('Actual vs Predicted Salaries')
        
        # Error distribution
        axes[0,1].hist(errors, bins=50, alpha=0.7)
        axes[0,1].set_xlabel('Prediction Error')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Prediction Error Distribution')
        
        # Residual plot
        axes[1,0].scatter(y_pred, errors, alpha=0.5)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Salary')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot')
        
        # Error by salary range
        salary_ranges = pd.cut(self.y_test, bins=5)
        error_by_range = pd.DataFrame({
            'Salary_Range': salary_ranges,
            'Error': errors
        }).groupby('Salary_Range')['Error'].mean()
        
        axes[1,1].bar(range(len(error_by_range)), error_by_range.values)
        axes[1,1].set_xlabel('Salary Range')
        axes[1,1].set_ylabel('Mean Error')
        axes[1,1].set_title('Mean Error by Salary Range')
        axes[1,1].set_xticks(range(len(error_by_range)))
        axes[1,1].set_xticklabels([str(x) for x in error_by_range.index], rotation=45)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return error_stats
    
    def salary_scenario_analysis(self):
        """Analyze salary predictions for different scenarios"""
        print("\n=== SALARY SCENARIO ANALYSIS ===")
        
        # Use best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        
        # Create scenario data
        scenarios = []
        
        # Scenario 1: Entry-level, On-site, Small company
        scenario1 = self.X.iloc[0].copy()
        scenario1['experience_level_encoded'] = 0  # Entry-level
        scenario1['remote_ratio'] = 0  # On-site
        scenario1['company_size_encoded'] = 0  # Small
        scenarios.append(('Entry-level, On-site, Small Company', scenario1))
        
        # Scenario 2: Senior-level, Remote, Large company
        scenario2 = self.X.iloc[0].copy()
        scenario2['experience_level_encoded'] = 2  # Senior-level
        scenario2['remote_ratio'] = 100  # Remote
        scenario2['company_size_encoded'] = 2  # Large
        scenarios.append(('Senior-level, Remote, Large Company', scenario2))
        
        # Scenario 3: Executive, Hybrid, Medium company
        scenario3 = self.X.iloc[0].copy()
        scenario3['experience_level_encoded'] = 3  # Executive
        scenario3['remote_ratio'] = 50  # Hybrid
        scenario3['company_size_encoded'] = 1  # Medium
        scenarios.append(('Executive, Hybrid, Medium Company', scenario3))
        
        # Make predictions
        scenario_results = []
        for name, scenario in scenarios:
            if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                scenario_scaled = self.scaler.transform(scenario.values.reshape(1, -1))
                prediction = best_model.predict(scenario_scaled)[0]
            else:
                prediction = best_model.predict(scenario.values.reshape(1, -1))[0]
            
            scenario_results.append({
                'Scenario': name,
                'Predicted Salary': f"${prediction:,.0f}"
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        print("Salary Predictions for Different Scenarios:")
        print(scenario_df.to_string(index=False))
        
        return scenario_df
    
    def generate_ml_report(self):
        """Generate comprehensive ML analysis report"""
        print("\n=== GENERATING MACHINE LEARNING REPORT ===")
        
        # Run all ML analyses
        results = self.train_models()
        comparison_df = self.model_comparison()
        feature_importance_df = self.feature_importance_analysis()
        error_stats = self.salary_prediction_insights()
        scenario_df = self.salary_scenario_analysis()
        
        # Create ML report
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model_metrics = results[best_model_name]
        
        report = f"""
        ========================================
        MACHINE LEARNING SALARY ANALYSIS REPORT
        ========================================
        
        MODEL PERFORMANCE:
        - Best Model: {best_model_name}
        - R² Score: {best_model_metrics['r2']:.4f}
        - RMSE: ${best_model_metrics['rmse']:,.0f}
        - MAE: ${best_model_metrics['mae']:,.0f}
        - Cross-Validation R²: {best_model_metrics['cv_mean']:.4f} (+/- {best_model_metrics['cv_std'] * 2:.4f})
        
        TOP 5 FEATURES BY IMPORTANCE:
        {feature_importance_df.head().to_string()}
        
        PREDICTION ERROR ANALYSIS:
        - Mean Error: ${error_stats['Mean Error']:,.0f}
        - Standard Deviation: ${error_stats['Std Error']:,.0f}
        - Mean Absolute Error: ${error_stats['Mean Absolute Error']:,.0f}
        
        SALARY SCENARIO PREDICTIONS:
        {scenario_df.to_string()}
        
        MODEL COMPARISON SUMMARY:
        {comparison_df.to_string()}
        
        VISUALIZATIONS GENERATED:
        - model_comparison.png
        - feature_importance.png
        - prediction_analysis.png
        
        KEY INSIGHTS:
        1. Model Performance: {best_model_name} achieved the best performance
        2. Feature Importance: {feature_importance_df.iloc[0]['Feature']} is the most important feature
        3. Prediction Accuracy: Model explains {best_model_metrics['r2']*100:.1f}% of salary variance
        4. Error Analysis: Average prediction error is ${error_stats['Mean Absolute Error']:,.0f}
        """
        
        # Save report
        with open('ml_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("ML Report saved as 'ml_analysis_report.txt'")
        print(report)
        
        return report

# Main execution
if __name__ == "__main__":
    print("Starting Machine Learning Salary Analysis...")
    
    # Initialize ML analyzer
    ml_analyzer = SalaryMLAnalyzer()
    
    # Generate comprehensive ML report
    report = ml_analyzer.generate_ml_report()
    
    print("\nMachine Learning analysis complete!")
