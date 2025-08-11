"""
BUS AN 512 Data Management and SQL - Group Project
Data Science Salary Analysis (2020-2025)
Python Analysis and Visualization Script

This script provides comprehensive analysis including:
- Data exploration and cleaning
- Statistical analysis and correlations
- Outlier detection
- Advanced visualizations
- Machine learning insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalaryDataAnalyzer:
    def __init__(self, file_path='salaries.csv'):
        """Initialize the analyzer with the salary dataset"""
        self.df = pd.read_csv(file_path)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        print("Cleaning and preparing data...")
        
        # Check for missing values
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        # Remove any rows with missing critical data
        self.df = self.df.dropna(subset=['salary_in_usd', 'work_year', 'experience_level'])
        
        # Convert work_year to datetime for better analysis
        self.df['work_year'] = pd.to_datetime(self.df['work_year'], format='%Y')
        
        # Create additional features
        self.df['salary_category'] = pd.cut(
            self.df['salary_in_usd'], 
            bins=[0, 50000, 100000, 150000, 200000, float('inf')],
            labels=['<50K', '50K-100K', '100K-150K', '150K-200K', '>200K']
        )
        
        # Create remote work categories
        self.df['remote_category'] = self.df['remote_ratio'].map({
            0: 'On-site',
            50: 'Hybrid', 
            100: 'Fully Remote'
        })
        
        print(f"Data shape after cleaning: {self.df.shape}")
        
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("\n=== BASIC STATISTICS ===")
        
        stats = {
            'Total Records': len(self.df),
            'Years Covered': self.df['work_year'].dt.year.nunique(),
            'Countries Represented': self.df['employee_residence'].nunique(),
            'Job Titles': self.df['job_title'].nunique(),
            'Average Salary (USD)': f"${self.df['salary_in_usd'].mean():,.0f}",
            'Median Salary (USD)': f"${self.df['salary_in_usd'].median():,.0f}",
            'Salary Std Dev': f"${self.df['salary_in_usd'].std():,.0f}",
            'Min Salary': f"${self.df['salary_in_usd'].min():,.0f}",
            'Max Salary': f"${self.df['salary_in_usd'].max():,.0f}"
        }
        
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        return stats
    
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Create correlation matrix for numerical variables
        numerical_cols = ['salary_in_usd', 'remote_ratio']
        correlation_matrix = self.df[numerical_cols].corr()
        
        print("Correlation Matrix:")
        print(correlation_matrix)
        
        # Visualize correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix - Numerical Variables')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def outlier_detection(self):
        """Detect and analyze outliers in salary data"""
        print("\n=== OUTLIER DETECTION ===")
        
        # Using IQR method for outlier detection
        Q1 = self.df['salary_in_usd'].quantile(0.25)
        Q3 = self.df['salary_in_usd'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df['salary_in_usd'] < lower_bound) | 
                          (self.df['salary_in_usd'] > upper_bound)]
        
        print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
        print(f"Lower bound: ${lower_bound:,.0f}")
        print(f"Upper bound: ${upper_bound:,.0f}")
        
        # Create outlier visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=self.df, y='salary_in_usd', ax=ax1)
        ax1.set_title('Salary Distribution with Outliers')
        ax1.set_ylabel('Salary (USD)')
        
        # Histogram with outlier highlighting
        ax2.hist(self.df['salary_in_usd'], bins=50, alpha=0.7, label='Normal Data')
        ax2.hist(outliers['salary_in_usd'], bins=50, alpha=0.7, color='red', label='Outliers')
        ax2.set_title('Salary Distribution - Outliers Highlighted')
        ax2.set_xlabel('Salary (USD)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outliers
    
    def salary_trends_visualization(self):
        """Create comprehensive salary trend visualizations"""
        print("\n=== SALARY TRENDS VISUALIZATION ===")
        
        # Prepare data for visualization
        yearly_stats = self.df.groupby(self.df['work_year'].dt.year).agg({
            'salary_in_usd': ['mean', 'median', 'std', 'count']
        }).round(0)
        yearly_stats.columns = ['Mean', 'Median', 'Std', 'Count']
        yearly_stats = yearly_stats.reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average salary trend
        axes[0,0].plot(yearly_stats['work_year'], yearly_stats['Mean'], marker='o', linewidth=2)
        axes[0,0].set_title('Average Salary Trend (2020-2025)')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Average Salary (USD)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Salary distribution by year
        for year in self.df['work_year'].dt.year.unique():
            year_data = self.df[self.df['work_year'].dt.year == year]['salary_in_usd']
            axes[0,1].hist(year_data, alpha=0.5, label=str(year), bins=30)
        axes[0,1].set_title('Salary Distribution by Year')
        axes[0,1].set_xlabel('Salary (USD)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # 3. Experience level salary comparison
        exp_salary = self.df.groupby('experience_level')['salary_in_usd'].mean().sort_values(ascending=False)
        axes[1,0].bar(exp_salary.index, exp_salary.values)
        axes[1,0].set_title('Average Salary by Experience Level')
        axes[1,0].set_xlabel('Experience Level')
        axes[1,0].set_ylabel('Average Salary (USD)')
        
        # 4. Remote work impact
        remote_salary = self.df.groupby('remote_category')['salary_in_usd'].mean()
        axes[1,1].bar(remote_salary.index, remote_salary.values)
        axes[1,1].set_title('Average Salary by Remote Work Category')
        axes[1,1].set_xlabel('Remote Work Category')
        axes[1,1].set_ylabel('Average Salary (USD)')
        
        plt.tight_layout()
        plt.savefig('salary_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return yearly_stats
    
    def geographic_analysis(self):
        """Analyze salary patterns by geography"""
        print("\n=== GEOGRAPHIC ANALYSIS ===")
        
        # Top countries by average salary
        country_stats = self.df.groupby('employee_residence').agg({
            'salary_in_usd': ['mean', 'count']
        }).round(0)
        country_stats.columns = ['Average_Salary', 'Record_Count']
        country_stats = country_stats.reset_index()
        
        # Filter for countries with sufficient data
        top_countries = country_stats[country_stats['Record_Count'] >= 50].nlargest(15, 'Average_Salary')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_countries['employee_residence'], top_countries['Average_Salary'])
        plt.title('Top 15 Countries by Average Salary (Min 50 records)')
        plt.xlabel('Average Salary (USD)')
        plt.ylabel('Country')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1000, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_countries
    
    def job_title_analysis(self):
        """Analyze salary patterns by job title"""
        print("\n=== JOB TITLE ANALYSIS ===")
        
        # Top job titles by average salary
        job_stats = self.df.groupby('job_title').agg({
            'salary_in_usd': ['mean', 'count', 'std']
        }).round(0)
        job_stats.columns = ['Average_Salary', 'Record_Count', 'Std_Dev']
        job_stats = job_stats.reset_index()
        
        # Filter for job titles with sufficient data
        top_jobs = job_stats[job_stats['Record_Count'] >= 20].nlargest(15, 'Average_Salary')
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        bars = plt.barh(top_jobs['job_title'], top_jobs['Average_Salary'])
        plt.title('Top 15 Job Titles by Average Salary (Min 20 records)')
        plt.xlabel('Average Salary (USD)')
        plt.ylabel('Job Title')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 2000, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('job_title_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_jobs
    
    def interactive_visualizations(self):
        """Create interactive visualizations using Plotly"""
        print("\n=== INTERACTIVE VISUALIZATIONS ===")
        
        # 1. Interactive salary trend
        yearly_data = self.df.groupby(self.df['work_year'].dt.year).agg({
            'salary_in_usd': ['mean', 'median', 'count']
        }).round(0)
        yearly_data.columns = ['Mean_Salary', 'Median_Salary', 'Record_Count']
        yearly_data = yearly_data.reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_data['work_year'], y=yearly_data['Mean_Salary'],
                                mode='lines+markers', name='Mean Salary',
                                line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=yearly_data['work_year'], y=yearly_data['Median_Salary'],
                                mode='lines+markers', name='Median Salary',
                                line=dict(color='red', width=3)))
        
        fig.update_layout(title='Interactive Salary Trends (2020-2025)',
                         xaxis_title='Year',
                         yaxis_title='Salary (USD)',
                         hovermode='x unified')
        
        fig.write_html('interactive_salary_trends.html')
        
        # 2. Interactive scatter plot: Experience vs Salary
        fig2 = px.scatter(self.df, x='experience_level', y='salary_in_usd',
                         color='remote_category', size='salary_in_usd',
                         hover_data=['job_title', 'employee_residence'],
                         title='Salary vs Experience Level by Remote Work Category')
        
        fig2.write_html('interactive_scatter.html')
        
        print("Interactive visualizations saved as HTML files")
    
    def statistical_insights(self):
        """Generate statistical insights and hypothesis testing"""
        print("\n=== STATISTICAL INSIGHTS ===")
        
        # 1. ANOVA test for experience levels
        from scipy import stats
        
        experience_groups = [group['salary_in_usd'].values for name, group in self.df.groupby('experience_level')]
        f_stat, p_value = stats.f_oneway(*experience_groups)
        
        print(f"ANOVA Test - Experience Level Impact on Salary:")
        print(f"F-statistic: {f_stat:.2f}")
        print(f"P-value: {p_value:.2e}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # 2. T-test for remote vs on-site
        remote_salaries = self.df[self.df['remote_ratio'] == 100]['salary_in_usd']
        onsite_salaries = self.df[self.df['remote_ratio'] == 0]['salary_in_usd']
        
        t_stat, p_value_remote = stats.ttest_ind(remote_salaries, onsite_salaries)
        
        print(f"\nT-test - Remote vs On-site Salaries:")
        print(f"T-statistic: {t_stat:.2f}")
        print(f"P-value: {p_value_remote:.2e}")
        print(f"Significant difference: {'Yes' if p_value_remote < 0.05 else 'No'}")
        
        # 3. Correlation with remote work
        correlation, p_value_corr = stats.pearsonr(self.df['salary_in_usd'], self.df['remote_ratio'])
        
        print(f"\nCorrelation - Salary vs Remote Work Ratio:")
        print(f"Correlation coefficient: {correlation:.3f}")
        print(f"P-value: {p_value_corr:.2e}")
        print(f"Significant correlation: {'Yes' if p_value_corr < 0.05 else 'No'}")
        
        return {
            'experience_anova': (f_stat, p_value),
            'remote_ttest': (t_stat, p_value_remote),
            'remote_correlation': (correlation, p_value_corr)
        }
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        correlation_matrix = self.correlation_analysis()
        outliers = self.outlier_detection()
        yearly_stats = self.salary_trends_visualization()
        top_countries = self.geographic_analysis()
        top_jobs = self.job_title_analysis()
        self.interactive_visualizations()
        statistical_results = self.statistical_insights()
        
        # Create summary report
        report = f"""
        ========================================
        DATA SCIENCE SALARY ANALYSIS REPORT
        ========================================
        
        DATASET OVERVIEW:
        - Total Records: {basic_stats['Total Records']:,}
        - Years Covered: {basic_stats['Years Covered']}
        - Countries: {basic_stats['Countries Represented']}
        - Job Titles: {basic_stats['Job Titles']}
        
        SALARY STATISTICS:
        - Average Salary: {basic_stats['Average Salary (USD)']}
        - Median Salary: {basic_stats['Median Salary (USD)']}
        - Salary Range: {basic_stats['Min Salary']} - {basic_stats['Max Salary']}
        
        KEY FINDINGS:
        1. Experience Level Impact: {statistical_results['experience_anova'][1]:.2e} (p-value)
        2. Remote Work Impact: {statistical_results['remote_ttest'][1]:.2e} (p-value)
        3. Remote Work Correlation: {statistical_results['remote_correlation'][0]:.3f}
        
        OUTLIERS:
        - Outliers Detected: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)
        
        TOP PAYING COUNTRIES:
        {top_countries.head(5).to_string()}
        
        TOP PAYING JOBS:
        {top_jobs.head(5).to_string()}
        
        VISUALIZATIONS GENERATED:
        - correlation_matrix.png
        - outlier_analysis.png
        - salary_trends.png
        - geographic_analysis.png
        - job_title_analysis.png
        - interactive_salary_trends.html
        - interactive_scatter.html
        """
        
        # Save report
        with open('salary_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("Report saved as 'salary_analysis_report.txt'")
        print(report)
        
        return report

# Main execution
if __name__ == "__main__":
    print("Starting Data Science Salary Analysis...")
    
    # Initialize analyzer
    analyzer = SalaryDataAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_report()
    
    print("\nAnalysis complete! Check the generated files for visualizations and insights.")
