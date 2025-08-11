"""
BUS AN 512 Data Management and SQL - Group Project
Data Science Salary Analysis (2020-2025)
Presentation Summary Script

This script generates key insights and visualizations specifically for the presentation.
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

# Set style for professional presentations
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class PresentationSummary:
    def __init__(self, file_path='salaries.csv'):
        """Initialize the presentation summary generator"""
        self.df = pd.read_csv(file_path)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare data for presentation"""
        print("Preparing data for presentation...")
        
        # Clean data
        self.df = self.df.dropna(subset=['salary_in_usd', 'work_year', 'experience_level'])
        
        # Create presentation-friendly categories
        self.df['experience_category'] = self.df['experience_level'].map({
            'EN': 'Entry-level',
            'MI': 'Mid-level', 
            'SE': 'Senior-level',
            'EX': 'Executive'
        })
        
        self.df['remote_category'] = self.df['remote_ratio'].map({
            0: 'On-site',
            50: 'Hybrid',
            100: 'Fully Remote'
        })
        
        self.df['company_size_category'] = self.df['company_size'].map({
            'S': 'Small (1-50)',
            'M': 'Medium (51-500)',
            'L': 'Large (500+)'
        })
        
        print(f"Data prepared: {len(self.df)} records")
        
    def generate_executive_summary(self):
        """Generate executive summary for presentation"""
        print("\n=== EXECUTIVE SUMMARY ===")
        
        # Key statistics
        total_records = len(self.df)
        avg_salary = self.df['salary_in_usd'].mean()
        median_salary = self.df['salary_in_usd'].median()
        countries = self.df['employee_residence'].nunique()
        job_titles = self.df['job_title'].nunique()
        years = self.df['work_year'].nunique()
        
        summary = f"""
        EXECUTIVE SUMMARY
        =================
        
        Dataset Overview:
        • {total_records:,} salary records from {countries} countries
        • {job_titles} unique job titles across {years} years (2020-2025)
        • Global average salary: ${avg_salary:,.0f}
        • Median salary: ${median_salary:,.0f}
        
        Key Insights:
        • Experience level is the strongest predictor of salary
        • Remote work adoption increased significantly post-COVID
        • Geographic salary disparities are substantial
        • Large companies pay higher salaries on average
        """
        
        print(summary)
        return summary
    
    def create_presentation_visualizations(self):
        """Create key visualizations for presentation"""
        print("\n=== CREATING PRESENTATION VISUALIZATIONS ===")
        
        # 1. Salary Distribution Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Salary distribution
        axes[0,0].hist(self.df['salary_in_usd'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        avg_salary = self.df['salary_in_usd'].mean()
        axes[0,0].axvline(avg_salary, color='red', linestyle='--', linewidth=2, label=f'Mean: ${avg_salary:,.0f}')
        axes[0,0].set_title('Salary Distribution (2020-2025)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Salary (USD)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Experience level impact
        exp_salary = self.df.groupby('experience_category')['salary_in_usd'].mean().sort_values(ascending=False)
        bars = axes[0,1].bar(exp_salary.index, exp_salary.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0,1].set_title('Average Salary by Experience Level', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Experience Level')
        axes[0,1].set_ylabel('Average Salary (USD)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 2000,
                          f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Remote work impact
        remote_salary = self.df.groupby('remote_category')['salary_in_usd'].mean()
        bars = axes[1,0].bar(remote_salary.index, remote_salary.values, color=['#FFEAA7', '#DDA0DD', '#98D8C8'])
        axes[1,0].set_title('Average Salary by Remote Work Category', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Remote Work Category')
        axes[1,0].set_ylabel('Average Salary (USD)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 2000,
                          f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Company size impact
        company_salary = self.df.groupby('company_size_category')['salary_in_usd'].mean()
        bars = axes[1,1].bar(company_salary.index, company_salary.values, color=['#F7DC6F', '#BB8FCE', '#85C1E9'])
        axes[1,1].set_title('Average Salary by Company Size', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Company Size')
        axes[1,1].set_ylabel('Average Salary (USD)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 2000,
                          f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('presentation_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Salary Trends Over Time
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall salary trend
        yearly_avg = self.df.groupby('work_year')['salary_in_usd'].mean()
        axes[0,0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, markersize=8)
        axes[0,0].set_title('Average Salary Trend (2020-2025)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Average Salary (USD)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(yearly_avg.index, yearly_avg.values):
            axes[0,0].text(x, y + 2000, f'${y:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Remote work adoption trend
        remote_trend = self.df.groupby(['work_year', 'remote_category']).size().unstack(fill_value=0)
        remote_trend_pct = remote_trend.div(remote_trend.sum(axis=1), axis=0) * 100
        
        remote_trend_pct.plot(kind='bar', stacked=True, ax=axes[0,1], 
                             color=['#FFEAA7', '#DDA0DD', '#98D8C8'])
        axes[0,1].set_title('Remote Work Adoption Trend', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Percentage (%)')
        axes[0,1].legend(title='Remote Work')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Experience level distribution
        exp_dist = self.df.groupby('experience_category').size()
        axes[1,0].pie(exp_dist.values, labels=exp_dist.index, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Distribution by Experience Level', fontsize=14, fontweight='bold')
        
        # Top countries by salary
        top_countries = self.df.groupby('employee_residence')['salary_in_usd'].mean().nlargest(10)
        bars = axes[1,1].barh(top_countries.index, top_countries.values)
        axes[1,1].set_title('Top 10 Countries by Average Salary', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Average Salary (USD)')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            axes[1,1].text(width + 2000, bar.get_y() + bar.get_height()/2,
                          f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('presentation_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Interactive Dashboard (Plotly)
        self.create_interactive_dashboard()
        
    def create_interactive_dashboard(self):
        """Create interactive dashboard for presentation"""
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Salary by Experience Level', 'Remote Work Impact', 
                          'Company Size Effect', 'Geographic Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Experience level
        exp_data = self.df.groupby('experience_category')['salary_in_usd'].mean()
        fig.add_trace(
            go.Bar(x=exp_data.index, y=exp_data.values, name="Experience Level",
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
            row=1, col=1
        )
        
        # 2. Remote work
        remote_data = self.df.groupby('remote_category')['salary_in_usd'].mean()
        fig.add_trace(
            go.Bar(x=remote_data.index, y=remote_data.values, name="Remote Work",
                  marker_color=['#FFEAA7', '#DDA0DD', '#98D8C8']),
            row=1, col=2
        )
        
        # 3. Company size
        company_data = self.df.groupby('company_size_category')['salary_in_usd'].mean()
        fig.add_trace(
            go.Bar(x=company_data.index, y=company_data.values, name="Company Size",
                  marker_color=['#F7DC6F', '#BB8FCE', '#85C1E9']),
            row=2, col=1
        )
        
        # 4. Geographic scatter
        country_data = self.df.groupby('employee_residence').agg({
            'salary_in_usd': 'mean',
            'remote_ratio': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=country_data['remote_ratio'], y=country_data['salary_in_usd'],
                      mode='markers', name="Countries",
                      text=country_data['employee_residence'],
                      marker=dict(size=10, color=country_data['salary_in_usd'],
                                colorscale='Viridis', showscale=True)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Data Science Salary Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Experience Level", row=1, col=1)
        fig.update_yaxes(title_text="Average Salary (USD)", row=1, col=1)
        fig.update_xaxes(title_text="Remote Work Category", row=1, col=2)
        fig.update_yaxes(title_text="Average Salary (USD)", row=1, col=2)
        fig.update_xaxes(title_text="Company Size", row=2, col=1)
        fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=1)
        fig.update_xaxes(title_text="Remote Work Ratio (%)", row=2, col=2)
        fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=2)
        
        fig.write_html('presentation_dashboard.html')
        print("Interactive dashboard saved as 'presentation_dashboard.html'")
        
    def generate_key_insights(self):
        """Generate key insights for presentation"""
        print("\n=== KEY INSIGHTS FOR PRESENTATION ===")
        
        # Calculate key insights
        insights = {}
        
        # 1. Experience level impact
        exp_impact = self.df.groupby('experience_category')['salary_in_usd'].mean()
        insights['experience_impact'] = {
            'highest': exp_impact.idxmax(),
            'lowest': exp_impact.idxmin(),
            'difference': exp_impact.max() - exp_impact.min(),
            'ratio': exp_impact.max() / exp_impact.min()
        }
        
        # 2. Remote work impact
        remote_impact = self.df.groupby('remote_category')['salary_in_usd'].mean()
        insights['remote_impact'] = {
            'highest': remote_impact.idxmax(),
            'lowest': remote_impact.idxmin(),
            'difference': remote_impact.max() - remote_impact.min()
        }
        
        # 3. Geographic disparity
        country_impact = self.df.groupby('employee_residence')['salary_in_usd'].mean()
        insights['geographic_disparity'] = {
            'highest_country': country_impact.idxmax(),
            'lowest_country': country_impact.idxmin(),
            'highest_salary': country_impact.max(),
            'lowest_salary': country_impact.min(),
            'ratio': country_impact.max() / country_impact.min()
        }
        
        # 4. Salary growth
        yearly_growth = self.df.groupby('work_year')['salary_in_usd'].mean()
        growth_rate = ((yearly_growth.iloc[-1] - yearly_growth.iloc[0]) / yearly_growth.iloc[0]) * 100
        insights['salary_growth'] = {
            'total_growth': growth_rate,
            'annual_growth': growth_rate / (len(yearly_growth) - 1)
        }
        
        # 5. Remote work adoption
        pre_covid = self.df[self.df['work_year'] <= 2020]['remote_ratio'].mean()
        post_covid = self.df[self.df['work_year'] >= 2023]['remote_ratio'].mean()
        insights['remote_adoption'] = {
            'pre_covid': pre_covid,
            'post_covid': post_covid,
            'increase': post_covid - pre_covid
        }
        
        # Print insights
        insights_text = f"""
        KEY INSIGHTS FOR PRESENTATION
        =============================
        
        1. EXPERIENCE LEVEL IMPACT:
        • {insights['experience_impact']['highest']} professionals earn ${insights['experience_impact']['difference']:,.0f} more than {insights['experience_impact']['lowest']}
        • Salary ratio between highest and lowest: {insights['experience_impact']['ratio']:.1f}x
        
        2. REMOTE WORK IMPACT:
        • {insights['remote_impact']['highest']} workers earn ${insights['remote_impact']['difference']:,.0f} more than {insights['remote_impact']['lowest']}
        
        3. GEOGRAPHIC DISPARITY:
        • {insights['geographic_disparity']['highest_country']} pays ${insights['geographic_disparity']['highest_salary']:,.0f} vs {insights['geographic_disparity']['lowest_country']} at ${insights['geographic_disparity']['lowest_salary']:,.0f}
        • Salary ratio: {insights['geographic_disparity']['ratio']:.1f}x
        
        4. SALARY GROWTH (2020-2025):
        • Total growth: {insights['salary_growth']['total_growth']:.1f}%
        • Annual growth: {insights['salary_growth']['annual_growth']:.1f}%
        
        5. REMOTE WORK ADOPTION:
        • Pre-COVID remote work: {insights['remote_adoption']['pre_covid']:.1f}%
        • Post-COVID remote work: {insights['remote_adoption']['post_covid']:.1f}%
        • Increase: {insights['remote_adoption']['increase']:.1f} percentage points
        """
        
        print(insights_text)
        return insights
    
    def generate_presentation_script(self):
        """Generate presentation script outline"""
        print("\n=== PRESENTATION SCRIPT OUTLINE ===")
        
        script = """
        PRESENTATION SCRIPT OUTLINE
        ===========================
        
        SLIDE 1: TITLE SLIDE
        - "Data Science Salary Analysis: Global Trends 2020-2025"
        - Group members and course information
        
        SLIDE 2: AGENDA
        - Dataset Overview
        - Key Findings
        - Salary Trends Analysis
        - Geographic Insights
        - Remote Work Impact
        - Machine Learning Insights
        - Conclusions & Recommendations
        
        SLIDE 3: DATASET OVERVIEW
        - 151,000+ salary records
        - 6 years of data (2020-2025)
        - Global coverage: [X] countries
        - [X] unique job titles
        - Average salary: $[X]
        
        SLIDE 4: KEY FINDINGS
        - Experience level is the strongest salary predictor
        - Remote work adoption increased significantly post-COVID
        - Geographic salary disparities are substantial
        - Large companies pay higher salaries
        
        SLIDE 5: SALARY TRENDS
        - Show salary trend chart
        - Discuss year-over-year growth
        - Highlight COVID-19 impact
        
        SLIDE 6: EXPERIENCE LEVEL IMPACT
        - Show experience level chart
        - Discuss salary progression
        - Executive vs Entry-level comparison
        
        SLIDE 7: REMOTE WORK ANALYSIS
        - Remote work adoption trend
        - Salary impact of remote work
        - Post-COVID changes
        
        SLIDE 8: GEOGRAPHIC INSIGHTS
        - Top paying countries
        - Geographic salary disparities
        - Regional trends
        
        SLIDE 9: MACHINE LEARNING INSIGHTS
        - Model performance summary
        - Feature importance
        - Prediction accuracy
        
        SLIDE 10: CONCLUSIONS
        - Summary of key insights
        - Business implications
        - Future trends
        
        SLIDE 11: Q&A
        - Thank you slide
        - Contact information
        """
        
        print(script)
        return script
    
    def generate_complete_presentation(self):
        """Generate complete presentation materials"""
        print("\n=== GENERATING COMPLETE PRESENTATION ===")
        
        # Generate all components
        summary = self.generate_executive_summary()
        self.create_presentation_visualizations()
        insights = self.generate_key_insights()
        script = self.generate_presentation_script()
        
        # Create comprehensive presentation file
        presentation_content = f"""
        ========================================
        BUS AN 512 PRESENTATION MATERIALS
        Data Science Salary Analysis (2020-2025)
        ========================================
        
        {summary}
        
        {insights}
        
        {script}
        
        FILES GENERATED:
        - presentation_overview.png
        - presentation_trends.png
        - presentation_dashboard.html
        
        PRESENTATION TIPS:
        1. Start with the executive summary
        2. Use the visualizations to support key points
        3. Emphasize the business value of insights
        4. Be prepared for questions about methodology
        5. Highlight the machine learning aspects
        6. Discuss practical applications of findings
        
        TIME ALLOCATION (11 minutes):
        - Introduction (1 min)
        - Dataset Overview (1 min)
        - Key Findings (2 min)
        - Visualizations & Trends (3 min)
        - Machine Learning Insights (2 min)
        - Conclusions (1 min)
        - Q&A Preparation (1 min)
        """
        
        # Save presentation materials
        with open('presentation_materials.txt', 'w') as f:
            f.write(presentation_content)
        
        print("Complete presentation materials saved as 'presentation_materials.txt'")
        print("\nPresentation materials ready! Check the generated files.")
        
        return presentation_content

# Main execution
if __name__ == "__main__":
    print("Generating BUS AN 512 Presentation Materials...")
    
    # Initialize presentation generator
    presenter = PresentationSummary()
    
    # Generate complete presentation
    materials = presenter.generate_complete_presentation()
    
    print("\nPresentation generation complete!")
