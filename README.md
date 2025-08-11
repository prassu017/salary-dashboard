# BUS AN 512 Data Management and SQL - Group Project
## Data Science Salary Analysis (2020-2025)

This project provides a comprehensive analysis of global Data Science, AI, and Machine Learning salary trends from 2020 to 2025. The analysis combines SQL queries, Python data analysis, machine learning, and advanced visualizations to extract valuable insights from over 151,000 salary records.

## 📊 Dataset Overview

- **Total Records**: 151,000+ salary entries
- **Time Period**: 2020-2025
- **Geographic Coverage**: Global (multiple countries)
- **Job Titles**: Data Science, AI, ML, and related roles
- **Key Variables**: Salary, experience level, remote work, company size, location

## 📁 Project Structure

```
archive/
├── salaries.csv                    # Main dataset
├── salaries.json                   # JSON version of dataset
├── 01_data_exploration.sql        # Basic SQL analysis and statistics
├── 02_trend_analysis.sql          # Time series and trend analysis
├── 03_advanced_analytics.sql      # Complex SQL insights and correlations
├── 04_presentation_insights.sql   # Key insights for presentation
├── 05_python_data_analysis.py     # Comprehensive Python analysis
├── 06_machine_learning_analysis.py # ML models and predictions
├── 07_presentation_summary.py     # Presentation materials generator
├── 08_dashboard_app.py            # Interactive Streamlit dashboard
├── run_dashboard.py               # Dashboard launcher script
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- SQL database (PostgreSQL, MySQL, SQLite, etc.)
- Required Python packages (see requirements.txt)

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your database:**
   - Import the `salaries.csv` file into your preferred SQL database
   - Ensure the table is named `salaries`

3. **Run the Interactive Dashboard (Recommended):**
   ```bash
   python run_dashboard.py
   ```
   This will automatically check dependencies and launch a comprehensive web-based dashboard with all insights and visualizations.

4. **Alternative: Run Individual Analyses:**

   **SQL Analysis:**
   - Execute the SQL files in order (01-04) in your database
   - Each file contains multiple queries with detailed comments

   **Python Analysis:**
   ```bash
   # Basic data analysis and visualizations
   python 05_python_data_analysis.py
   
   # Machine learning analysis
   python 06_machine_learning_analysis.py
   
   # Generate presentation materials
   python 07_presentation_summary.py
   ```

## 📈 Analysis Components

### 1. SQL Analysis (`01_data_exploration.sql` - `04_presentation_insights.sql`)

**File 01: Data Exploration**
- Basic dataset statistics
- Salary distribution analysis
- Experience level impact
- Employment type distribution
- Company size analysis
- Remote work patterns
- Top job titles and countries

**File 02: Trend Analysis**
- Salary growth trends by year and experience
- Remote work adoption over time
- Company size distribution trends
- Job title evolution (2020 vs 2025)
- Geographic salary trends
- Employment type trends

**File 03: Advanced Analytics**
- Salary premium analysis (remote vs on-site)
- Company size vs experience level analysis
- Geographic salary disparity analysis
- Job title salary distribution
- Employment type vs company size
- Salary growth rate analysis
- Remote work adoption by country

**File 04: Presentation Insights**
- Executive summary statistics
- Top 5 key findings
- COVID-19 impact analysis
- Emerging job titles
- Salary prediction insights
- Business intelligence summary

### 2. Python Analysis (`05_python_data_analysis.py`)

**Features:**
- Data cleaning and preparation
- Statistical analysis and correlations
- Outlier detection using IQR method
- Advanced visualizations (matplotlib, seaborn)
- Interactive visualizations (plotly)
- Statistical hypothesis testing
- Comprehensive reporting

**Generated Files:**
- `correlation_matrix.png`
- `outlier_analysis.png`
- `salary_trends.png`
- `geographic_analysis.png`
- `job_title_analysis.png`
- `interactive_salary_trends.html`
- `interactive_scatter.html`
- `salary_analysis_report.txt`

### 3. Machine Learning Analysis (`06_machine_learning_analysis.py`)

**Models Implemented:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Decision Tree

**Features:**
- Model performance comparison
- Feature importance analysis
- Salary prediction insights
- Scenario analysis
- Cross-validation
- Error analysis

**Generated Files:**
- `model_comparison.png`
- `feature_importance.png`
- `prediction_analysis.png`
- `ml_analysis_report.txt`

### 4. Presentation Materials (`07_presentation_summary.py`)

**Features:**
- Executive summary generation
- Professional visualizations
- Interactive dashboard
- Key insights calculation
- Presentation script outline
- Time allocation guidance

**Generated Files:**
- `presentation_overview.png`
- `presentation_trends.png`
- `presentation_dashboard.html`
- `presentation_materials.txt`

### 5. Interactive Dashboard (`08_dashboard_app.py`)

**Features:**
- **Overview Section**: Executive summary with key metrics and insights
- **Data Explorer**: Interactive filtering and data exploration
- **Visualizations**: Comprehensive charts and graphs
- **Advanced Analytics**: Correlation analysis and statistical insights
- **Machine Learning**: Feature importance and salary prediction tool
- **Raw Data**: Grid view of the complete dataset with download option

**Dashboard Sections:**
- 📊 Executive Summary with key metrics
- 📈 Interactive Data Explorer with filters
- 📊 Comprehensive Visualizations
- 🔍 Advanced Analytics and correlations
- 🤖 Machine Learning insights and predictions
- 📋 Raw Data grid with export functionality

**Access:** Run `python run_dashboard.py` and open http://localhost:8501 in your browser

## 🎯 Key Insights

### 1. Experience Level Impact
- Executive-level professionals earn significantly more than entry-level
- Clear salary progression from entry to executive levels
- Experience is the strongest predictor of salary

### 2. Remote Work Trends
- Significant increase in remote work adoption post-COVID
- Remote work patterns vary by country and company size
- Salary impact of remote work varies by experience level

### 3. Geographic Disparities
- Substantial salary differences between countries
- Top-paying countries show clear patterns
- Regional trends in remote work adoption

### 4. Company Size Effect
- Large companies generally pay higher salaries
- Different experience level distributions by company size
- Remote work adoption varies by company size

### 5. Salary Growth Trends
- Overall salary growth from 2020-2025
- Year-over-year growth patterns
- Impact of economic factors

## 📊 Visualizations

The project generates various types of visualizations:

1. **Static Charts (PNG):**
   - Salary distribution histograms
   - Bar charts for categorical comparisons
   - Trend lines for time series
   - Correlation matrices
   - Box plots for outlier analysis

2. **Interactive Dashboards (HTML):**
   - Multi-panel dashboards
   - Hover information
   - Zoom and pan capabilities
   - Filtering options

3. **Machine Learning Visualizations:**
   - Model performance comparisons
   - Feature importance charts
   - Prediction vs actual plots
   - Residual analysis

## 🔧 Customization

### Modifying SQL Queries
- Each SQL file contains multiple queries
- Queries are well-commented and organized
- Modify parameters (e.g., year ranges, thresholds) as needed

### Adjusting Python Analysis
- Modify the `SalaryDataAnalyzer` class methods
- Add new visualization types
- Customize statistical tests
- Adjust outlier detection parameters

### Machine Learning Customization
- Add new models to the `models` dictionary
- Modify feature engineering
- Adjust hyperparameters
- Add new evaluation metrics

## 📝 Presentation Guidelines

### Time Allocation (11 minutes total)
- Introduction (1 min)
- Dataset Overview (1 min)
- Key Findings (2 min)
- Visualizations & Trends (3 min)
- Machine Learning Insights (2 min)
- Conclusions (1 min)
- Q&A Preparation (1 min)

### Key Points to Emphasize
1. **Business Value**: How insights can inform hiring and compensation decisions
2. **Methodology**: Explain the analytical approach and data quality
3. **Trends**: Highlight significant patterns and changes over time
4. **Predictions**: Discuss machine learning model performance and applications
5. **Limitations**: Acknowledge data limitations and assumptions

## 🤝 Group Collaboration

### File Organization
- Each team member can work on different analysis components
- SQL files can be executed independently
- Python scripts can be run separately
- Generated files are clearly named and organized

### Version Control
- Use Git for collaborative development
- Commit analysis results and visualizations
- Document any customizations or modifications

## 📚 Additional Resources

### Data Sources
- Original dataset: AIJobs salary survey (CC0 license)
- Additional sources: 365DataScience, Payscale, KDnuggets, ZipRecruiter

### Technical Documentation
- SQL documentation for your specific database
- Python package documentation (pandas, matplotlib, scikit-learn)
- Statistical analysis references

### Presentation Resources
- Business casual attire requirement
- Professional presentation templates
- Q&A preparation materials

## 🎓 Academic Requirements

This project satisfies the BUS AN 512 requirements:
- ✅ Dataset approval (due 7/30)
- ✅ SQL analysis and data management
- ✅ Data cleaning and importing
- ✅ Visualization and insights
- ✅ 11-minute presentation (due 8/16)
- ✅ Business casual presentation

## 📞 Support

For questions or issues:
1. Check the generated reports for detailed explanations
2. Review the SQL comments for query explanations
3. Examine the Python code comments for analysis details
4. Consult the presentation materials for guidance

---

**Good luck with your presentation! 🚀**
