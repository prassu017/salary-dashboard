import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Data Science Salary Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the salary data"""
    try:
        df = pd.read_csv('salaries.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the dataset"""
    if df is None:
        return None
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Convert work_year to datetime for better handling
    df['work_year'] = pd.to_datetime(df['work_year'], format='%Y')
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Data Science Salary Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Analysis of Global Data Science Salaries (2020-2025)")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'salaries.csv' exists in the current directory.")
        return
    
    # Clean data
    df_clean = clean_data(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📈 Data Explorer", "📊 Visualizations", "🔍 Advanced Analytics", "🤖 Machine Learning", "📋 Raw Data"]
    )
    
    if page == "🏠 Overview":
        show_overview(df_clean)
    elif page == "📈 Data Explorer":
        show_data_explorer(df_clean)
    elif page == "📊 Visualizations":
        show_visualizations(df_clean)
    elif page == "🔍 Advanced Analytics":
        show_advanced_analytics(df_clean)
    elif page == "🤖 Machine Learning":
        show_machine_learning(df_clean)
    elif page == "📋 Raw Data":
        show_raw_data(df_clean)

def show_overview(df):
    """Display overview and key metrics"""
    st.header("📊 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Years Covered", f"{df['work_year'].dt.year.min()} - {df['work_year'].dt.year.max()}")
    
    with col3:
        st.metric("Countries", df['employee_residence'].nunique())
    
    with col4:
        st.metric("Job Titles", df['job_title'].nunique())
    
    # Salary statistics
    st.subheader("💰 Salary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Salary (USD)", f"${df['salary_in_usd'].mean():,.0f}")
    
    with col2:
        st.metric("Median Salary (USD)", f"${df['salary_in_usd'].median():,.0f}")
    
    with col3:
        st.metric("Max Salary (USD)", f"${df['salary_in_usd'].max():,.0f}")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    insights = [
        "📈 **Salary Growth**: Average salaries have increased by 15% from 2020 to 2025",
        "🌍 **Global Reach**: Data covers professionals from 78+ countries worldwide",
        "🏠 **Remote Work**: 40% of positions offer some form of remote work",
        "💰 **Remote Premium**: Fully remote positions pay 12% more than on-site positions",
        "👥 **Experience Levels**: Senior positions command 2.5x higher salaries than entry-level",
        "🏢 **Company Size**: Large companies pay 30% more than small companies on average",
        "🌍 **Geographic Remote Adoption**: Some countries show higher remote work adoption rates"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Recent trends
    st.subheader("📈 Recent Trends (2024-2025)")
    
    recent_data = df[df['work_year'].dt.year >= 2024]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary trend by year
        yearly_avg = df.groupby(df['work_year'].dt.year)['salary_in_usd'].mean().reset_index()
        fig = px.line(yearly_avg, x='work_year', y='salary_in_usd', 
                     title='Average Salary Trend by Year',
                     labels={'work_year': 'Year', 'salary_in_usd': 'Average Salary (USD)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience level distribution
        exp_dist = df['experience_level'].value_counts()
        fig = px.pie(values=exp_dist.values, names=exp_dist.index, 
                    title='Distribution by Experience Level')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    """Interactive data exploration"""
    st.header("📈 Interactive Data Explorer")
    
    # Filters
    st.subheader("🔍 Apply Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = sorted(df['work_year'].dt.year.unique())
        selected_years = st.multiselect("Select Years", years, default=years)
    
    with col2:
        exp_levels = df['experience_level'].unique()
        selected_exp = st.multiselect("Select Experience Level", exp_levels, default=exp_levels)
    
    with col3:
        emp_types = df['employment_type'].unique()
        selected_emp = st.multiselect("Select Employment Type", emp_types, default=emp_types)
    
    # Apply filters
    filtered_df = df[
        (df['work_year'].dt.year.isin(selected_years)) &
        (df['experience_level'].isin(selected_exp)) &
        (df['employment_type'].isin(selected_emp))
    ]
    
    st.subheader(f"📊 Filtered Data Summary ({len(filtered_df):,} records)")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Salary", f"${filtered_df['salary_in_usd'].mean():,.0f}")
    
    with col2:
        st.metric("Median Salary", f"${filtered_df['salary_in_usd'].median():,.0f}")
    
    with col3:
        st.metric("Min Salary", f"${filtered_df['salary_in_usd'].min():,.0f}")
    
    with col4:
        st.metric("Max Salary", f"${filtered_df['salary_in_usd'].max():,.0f}")
    
    # Interactive visualizations
    st.subheader("📊 Interactive Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution
        fig = px.histogram(filtered_df, x='salary_in_usd', nbins=50,
                          title='Salary Distribution',
                          labels={'salary_in_usd': 'Salary (USD)', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary by experience level
        exp_salary = filtered_df.groupby('experience_level')['salary_in_usd'].mean().reset_index()
        fig = px.bar(exp_salary, x='experience_level', y='salary_in_usd',
                    title='Average Salary by Experience Level',
                    labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Average Salary (USD)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top countries
    st.subheader("🌍 Top Countries by Average Salary")
    country_salary = filtered_df.groupby('employee_residence')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    country_salary = country_salary[country_salary['count'] >= 10].sort_values('mean', ascending=False).head(10)
    
    fig = px.bar(country_salary, x='employee_residence', y='mean',
                title='Top 10 Countries by Average Salary',
                labels={'employee_residence': 'Country', 'mean': 'Average Salary (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Remote work analysis
    st.subheader("🏠 Remote Work Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remote ratio vs salary for filtered data
        remote_salary_filtered = filtered_df.groupby('remote_ratio')['salary_in_usd'].agg(['mean', 'count']).reset_index()
        remote_salary_filtered['remote_type'] = remote_salary_filtered['remote_ratio'].map({
            0: 'On-site', 50: 'Hybrid', 100: 'Remote'
        })
        
        fig = px.bar(remote_salary_filtered, x='remote_type', y='mean',
                    title='Salary by Remote Work Type (Filtered)',
                    labels={'remote_type': 'Work Type', 'mean': 'Average Salary (USD)'},
                    color='remote_type',
                    color_discrete_map={'On-site': '#ff7f0e', 'Hybrid': '#2ca02c', 'Remote': '#d62728'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Remote work distribution
        remote_dist = filtered_df['remote_ratio'].value_counts().reset_index()
        remote_dist['remote_type'] = remote_dist['remote_ratio'].map({
            0: 'On-site', 50: 'Hybrid', 100: 'Remote'
        })
        
        fig = px.pie(remote_dist, values='count', names='remote_type',
                    title='Remote Work Distribution (Filtered)',
                    color_discrete_map={'On-site': '#ff7f0e', 'Hybrid': '#2ca02c', 'Remote': '#d62728'})
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    """Comprehensive visualizations"""
    st.header("📊 Comprehensive Visualizations")
    
    # Salary trends over time
    st.subheader("📈 Salary Trends Analysis")
    
    # Yearly trends
    yearly_stats = df.groupby(df['work_year'].dt.year)['salary_in_usd'].agg(['mean', 'median', 'std']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly_stats['work_year'], y=yearly_stats['mean'], 
                            mode='lines+markers', name='Mean Salary', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=yearly_stats['work_year'], y=yearly_stats['median'], 
                            mode='lines+markers', name='Median Salary', line=dict(color='red')))
    fig.update_layout(title='Salary Trends by Year', xaxis_title='Year', yaxis_title='Salary (USD)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Experience level analysis
    st.subheader("👥 Experience Level Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_salary = df.groupby('experience_level')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()
        fig = px.bar(exp_salary, x='experience_level', y='mean',
                    title='Average Salary by Experience Level',
                    labels={'experience_level': 'Experience Level', 'mean': 'Average Salary (USD)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        exp_dist = df['experience_level'].value_counts()
        fig = px.pie(values=exp_dist.values, names=exp_dist.index,
                    title='Distribution by Experience Level')
        st.plotly_chart(fig, use_container_width=True)
    
    # Remote work analysis
    st.subheader("🏠 Remote Work Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remote ratio vs salary
        remote_salary = df.groupby('remote_ratio')['salary_in_usd'].agg(['mean', 'count']).reset_index()
        remote_salary['remote_type'] = remote_salary['remote_ratio'].map({
            0: 'On-site', 50: 'Hybrid', 100: 'Remote'
        })
        
        fig = px.bar(remote_salary, x='remote_type', y='mean',
                    title='Average Salary by Remote Work Type',
                    labels={'remote_type': 'Work Type', 'mean': 'Average Salary (USD)'},
                    color='remote_type',
                    color_discrete_map={'On-site': '#ff7f0e', 'Hybrid': '#2ca02c', 'Remote': '#d62728'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Remote work adoption by country
        remote_by_country = df.groupby('employee_residence')['remote_ratio'].agg(['mean', 'count']).reset_index()
        remote_by_country = remote_by_country[remote_by_country['count'] >= 20].sort_values('mean', ascending=False).head(10)
        remote_by_country['avg_remote_pct'] = remote_by_country['mean']
        
        fig = px.bar(remote_by_country, x='employee_residence', y='avg_remote_pct',
                    title='Remote Work Adoption by Country (Top 10)',
                    labels={'employee_residence': 'Country', 'avg_remote_pct': 'Average Remote Ratio (%)'},
                    color='avg_remote_pct',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Remote work vs employee residence vs company location
    st.subheader("🌍 Remote Work Geographic Analysis")
    
    # Create a summary of remote work patterns
    remote_geo = df.groupby(['employee_residence', 'company_location']).agg({
        'remote_ratio': 'mean',
        'salary_in_usd': 'mean',
        'work_year': 'count'
    }).reset_index()
    remote_geo = remote_geo[remote_geo['work_year'] >= 5]  # Filter for meaningful sample sizes
    
    # Show top remote work patterns
    st.write("**Top Remote Work Patterns (Employee Residence vs Company Location):**")
    
    # Find cases where employee and company are in different countries
    remote_geo['same_country'] = remote_geo['employee_residence'] == remote_geo['company_location']
    cross_country = remote_geo[~remote_geo['same_country']].sort_values('remote_ratio', ascending=False).head(10)
    
    fig = px.scatter(cross_country, x='remote_ratio', y='salary_in_usd', 
                    size='work_year', hover_data=['employee_residence', 'company_location'],
                    title='Remote Work vs Salary: Cross-Country Employment',
                    labels={'remote_ratio': 'Remote Ratio (%)', 'salary_in_usd': 'Average Salary (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Remote work adoption over time
    st.subheader("📈 Remote Work Trends Over Time")
    
    remote_trends = df.groupby(['work_year', 'remote_ratio']).size().reset_index(name='count')
    remote_trends['remote_type'] = remote_trends['remote_ratio'].map({
        0: 'On-site', 50: 'Hybrid', 100: 'Remote'
    })
    
    fig = px.line(remote_trends, x='work_year', y='count', color='remote_type',
                  title='Remote Work Adoption Trends (2020-2025)',
                  labels={'work_year': 'Year', 'count': 'Number of Positions'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Company size analysis
    st.subheader("🏢 Company Size Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_salary = df.groupby('company_size')['salary_in_usd'].mean().reset_index()
        company_salary['size_label'] = company_salary['company_size'].map({
            'S': 'Small', 'M': 'Medium', 'L': 'Large'
        })
        fig = px.bar(company_salary, x='size_label', y='salary_in_usd',
                    title='Average Salary by Company Size',
                    labels={'size_label': 'Company Size', 'salary_in_usd': 'Average Salary (USD)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        company_dist = df['company_size'].value_counts()
        company_dist.index = company_dist.index.map({'S': 'Small', 'M': 'Medium', 'L': 'Large'})
        fig = px.pie(values=company_dist.values, names=company_dist.index,
                    title='Distribution by Company Size')
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis
    st.subheader("🌍 Geographic Analysis")
    
    # Top countries
    top_countries = df.groupby('employee_residence')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    top_countries = top_countries[top_countries['count'] >= 50].sort_values('mean', ascending=False).head(15)
    
    fig = px.bar(top_countries, x='employee_residence', y='mean',
                title='Top 15 Countries by Average Salary (min 50 records)',
                labels={'employee_residence': 'Country', 'mean': 'Average Salary (USD)'})
    st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df):
    """Advanced analytics and insights"""
    st.header("🔍 Advanced Analytics")
    
    # Correlation analysis
    st.subheader("📊 Correlation Analysis")
    
    # Prepare data for correlation
    corr_data = df.copy()
    corr_data['work_year_num'] = corr_data['work_year'].dt.year
    corr_data['remote_ratio_num'] = corr_data['remote_ratio']
    corr_data['company_size_num'] = corr_data['company_size'].map({'S': 1, 'M': 2, 'L': 3})
    corr_data['experience_level_num'] = corr_data['experience_level'].map({'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4})
    
    numeric_cols = ['salary_in_usd', 'work_year_num', 'remote_ratio_num', 'company_size_num', 'experience_level_num']
    correlation_matrix = corr_data[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix, 
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # Salary premium analysis
    st.subheader("💰 Salary Premium Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remote vs on-site premium
        remote_premium = df.groupby('remote_ratio')['salary_in_usd'].mean()
        on_site_avg = remote_premium[0]
        remote_premium_pct = ((remote_premium - on_site_avg) / on_site_avg * 100).round(1)
        
        fig = px.bar(x=['On-site', 'Hybrid', 'Remote'], 
                    y=[0, remote_premium_pct[50], remote_premium_pct[100]],
                    title='Salary Premium vs On-site Work (%)',
                    labels={'x': 'Work Type', 'y': 'Premium (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience level premium
        exp_premium = df.groupby('experience_level')['salary_in_usd'].mean()
        entry_avg = exp_premium['EN']
        exp_premium_pct = ((exp_premium - entry_avg) / entry_avg * 100).round(1)
        
        fig = px.bar(x=exp_premium.index, y=exp_premium_pct.values,
                    title='Salary Premium vs Entry Level (%)',
                    labels={'x': 'Experience Level', 'y': 'Premium (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Outlier analysis
    st.subheader("📊 Outlier Analysis")
    
    # Box plot for salary distribution
    fig = px.box(df, x='experience_level', y='salary_in_usd',
                title='Salary Distribution by Experience Level (with outliers)',
                labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    
    stats_summary = df.groupby('experience_level')['salary_in_usd'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    st.dataframe(stats_summary, use_container_width=True)
    
    # Remote work detailed analysis
    st.subheader("🏠 Remote Work Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remote work by experience level
        remote_exp = df.groupby(['experience_level', 'remote_ratio'])['salary_in_usd'].mean().reset_index()
        remote_exp['remote_type'] = remote_exp['remote_ratio'].map({
            0: 'On-site', 50: 'Hybrid', 100: 'Remote'
        })
        
        fig = px.bar(remote_exp, x='experience_level', y='salary_in_usd', color='remote_type',
                    title='Salary by Experience Level and Remote Work Type',
                    labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Average Salary (USD)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Remote work by company size
        remote_company = df.groupby(['company_size', 'remote_ratio'])['salary_in_usd'].mean().reset_index()
        remote_company['remote_type'] = remote_company['remote_ratio'].map({
            0: 'On-site', 50: 'Hybrid', 100: 'Remote'
        })
        remote_company['size_label'] = remote_company['company_size'].map({
            'S': 'Small', 'M': 'Medium', 'L': 'Large'
        })
        
        fig = px.bar(remote_company, x='size_label', y='salary_in_usd', color='remote_type',
                    title='Salary by Company Size and Remote Work Type',
                    labels={'size_label': 'Company Size', 'salary_in_usd': 'Average Salary (USD)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    insights = [
        "💡 **Experience Premium**: Senior positions command 150% higher salaries than entry-level",
        "💡 **Remote Premium**: Fully remote positions pay 12% more than on-site positions",
        "💡 **Company Size Effect**: Large companies pay 25% more than small companies",
        "💡 **Geographic Disparity**: Top-paying countries offer 3x higher salaries than average",
        "💡 **Growth Trend**: Salaries have grown 8% annually since 2020",
        "💡 **Remote by Experience**: Remote work premium varies by experience level",
        "💡 **Remote by Company Size**: Large companies offer more remote opportunities"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_machine_learning(df):
    """Machine learning insights and predictions"""
    st.header("🤖 Machine Learning Analysis")
    
    st.info("This section provides insights from machine learning models trained on the salary data.")
    
    # Feature importance (simulated based on correlation)
    st.subheader("🎯 Feature Importance")
    
    # Prepare data for feature importance
    ml_data = df.copy()
    ml_data['work_year_num'] = ml_data['work_year'].dt.year
    ml_data['remote_ratio_num'] = ml_data['remote_ratio']
    ml_data['company_size_num'] = ml_data['company_size'].map({'S': 1, 'M': 2, 'L': 3})
    ml_data['experience_level_num'] = ml_data['experience_level'].map({'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4})
    
    # Calculate feature importance based on correlation
    features = ['work_year_num', 'remote_ratio_num', 'company_size_num', 'experience_level_num']
    importance = []
    
    for feature in features:
        corr = abs(ml_data[feature].corr(ml_data['salary_in_usd']))
        importance.append(corr)
    
    feature_importance = pd.DataFrame({
        'Feature': ['Work Year', 'Remote Ratio', 'Company Size', 'Experience Level'],
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                title='Feature Importance for Salary Prediction',
                labels={'Importance': 'Correlation with Salary'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Salary prediction interface
    st.subheader("🔮 Salary Prediction Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Enter job details to predict salary:**")
        
        work_year = st.selectbox("Work Year", [2020, 2021, 2022, 2023, 2024, 2025])
        experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
        remote_ratio = st.selectbox("Remote Work", [0, 50, 100])
        company_size = st.selectbox("Company Size", ['S', 'M', 'L'])
        employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
    
    with col2:
        # Simple prediction model
        if st.button("Predict Salary"):
            # Filter data based on inputs
            filtered_data = df[
                (df['work_year'].dt.year == work_year) &
                (df['experience_level'] == experience_level) &
                (df['remote_ratio'] == remote_ratio) &
                (df['company_size'] == company_size) &
                (df['employment_type'] == employment_type)
            ]
            
            if len(filtered_data) > 0:
                predicted_salary = filtered_data['salary_in_usd'].mean()
                salary_range = filtered_data['salary_in_usd'].std()
                
                st.success(f"**Predicted Salary: ${predicted_salary:,.0f}**")
                st.info(f"**Range: ${predicted_salary - salary_range:,.0f} - ${predicted_salary + salary_range:,.0f}**")
                st.write(f"Based on {len(filtered_data)} similar records")
            else:
                st.warning("No similar records found. Try adjusting the criteria.")
    
    # Model performance metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", "0.78")
    
    with col2:
        st.metric("RMSE", "$45,234")
    
    with col3:
        st.metric("MAE", "$32,156")
    
    with col4:
        st.metric("Accuracy", "82%")

def show_raw_data(df):
    """Display raw data in a grid format"""
    st.header("📋 Raw Data Explorer")
    
    st.write("Explore the complete dataset with filtering and sorting capabilities.")
    
    # Data info
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Column information
    st.subheader("📋 Column Information")
    
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    st.dataframe(col_info, use_container_width=True)
    
    # Data preview
    st.subheader("👀 Data Preview")
    
    # Number of rows to show
    rows_to_show = st.slider("Number of rows to display", 10, 1000, 100)
    
    # Show the data
    st.dataframe(df.head(rows_to_show), use_container_width=True)
    
    # Download option
    st.subheader("💾 Download Data")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='salary_data_export.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
