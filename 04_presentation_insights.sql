-- BUS AN 512 Data Management and SQL - Group Project
-- Data Science Salary Analysis (2020-2025)
-- File 4: Key Insights for Presentation

-- 1. Executive Summary - Key Statistics
SELECT 
    'Dataset Overview' as metric,
    COUNT(*) as value,
    'Total salary records' as description
FROM salaries
UNION ALL
SELECT 
    'Time Period',
    MAX(work_year) - MIN(work_year) + 1,
    'Years of data coverage'
FROM salaries
UNION ALL
SELECT 
    'Global Reach',
    COUNT(DISTINCT employee_residence),
    'Countries represented'
FROM salaries
UNION ALL
SELECT 
    'Job Diversity',
    COUNT(DISTINCT job_title),
    'Unique job titles'
FROM salaries
UNION ALL
SELECT 
    'Average Salary (USD)',
    ROUND(AVG(salary_in_usd), 0),
    'Global average annual salary'
FROM salaries;

-- 2. Top 5 Key Findings for Presentation
-- Finding 1: Salary Growth Over Time
SELECT 
    'Finding 1: Salary Growth Trend' as insight,
    work_year,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(
        (AVG(salary_in_usd) - LAG(AVG(salary_in_usd)) OVER (ORDER BY work_year)) / 
        LAG(AVG(salary_in_usd)) OVER (ORDER BY work_year) * 100, 1
    ) as year_over_year_growth_percent
FROM salaries
GROUP BY work_year
ORDER BY work_year;

-- Finding 2: Experience Level Impact
SELECT 
    'Finding 2: Experience Level Premium' as insight,
    experience_level,
    CASE 
        WHEN experience_level = 'EN' THEN 'Entry-level'
        WHEN experience_level = 'MI' THEN 'Mid-level'
        WHEN experience_level = 'SE' THEN 'Senior-level'
        WHEN experience_level = 'EX' THEN 'Executive'
    END as experience_description,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(
        (AVG(salary_in_usd) - (SELECT AVG(salary_in_usd) FROM salaries)) / 
        (SELECT AVG(salary_in_usd) FROM salaries) * 100, 1
    ) as premium_vs_global_avg_percent
FROM salaries
GROUP BY experience_level
ORDER BY avg_salary_usd DESC;

-- Finding 3: Remote Work Impact
SELECT 
    'Finding 3: Remote Work Premium' as insight,
    remote_ratio,
    CASE 
        WHEN remote_ratio = 0 THEN 'On-site'
        WHEN remote_ratio = 50 THEN 'Hybrid'
        WHEN remote_ratio = 100 THEN 'Fully Remote'
    END as work_setting,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM salaries), 1) as percentage_of_workforce
FROM salaries
GROUP BY remote_ratio
ORDER BY remote_ratio;

-- Finding 4: Company Size Impact
SELECT 
    'Finding 4: Company Size Effect' as insight,
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50)'
        WHEN company_size = 'M' THEN 'Medium (51-500)'
        WHEN company_size = 'L' THEN 'Large (501+)'
    END as company_size_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM salaries), 1) as percentage_of_workforce
FROM salaries
GROUP BY company_size
ORDER BY avg_salary_usd DESC;

-- Finding 5: Geographic Disparity
SELECT 
    'Finding 5: Geographic Salary Disparity' as insight,
    employee_residence,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(
        (AVG(salary_in_usd) - (SELECT AVG(salary_in_usd) FROM salaries)) / 
        (SELECT AVG(salary_in_usd) FROM salaries) * 100, 1
    ) as deviation_from_global_avg_percent
FROM salaries
GROUP BY employee_residence
HAVING COUNT(*) >= 100  -- Only countries with significant data
ORDER BY avg_salary_usd DESC
LIMIT 10;

-- 3. COVID-19 Impact Analysis (2020-2022 vs 2023-2025)
WITH covid_periods AS (
    SELECT 
        CASE 
            WHEN work_year IN (2020, 2021, 2022) THEN 'COVID Period (2020-2022)'
            WHEN work_year IN (2023, 2024, 2025) THEN 'Post-COVID Period (2023-2025)'
        END as period,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
        ROUND(AVG(CASE WHEN remote_ratio = 100 THEN 1 ELSE 0 END) * 100, 1) as fully_remote_percentage
    FROM salaries
    WHERE work_year >= 2020
    GROUP BY 
        CASE 
            WHEN work_year IN (2020, 2021, 2022) THEN 'COVID Period (2020-2022)'
            WHEN work_year IN (2023, 2024, 2025) THEN 'Post-COVID Period (2023-2025)'
        END
)
SELECT 
    'COVID-19 Impact Analysis' as analysis_type,
    period,
    record_count,
    avg_salary_usd,
    fully_remote_percentage,
    ROUND(
        (avg_salary_usd - LAG(avg_salary_usd) OVER (ORDER BY period)) / 
        LAG(avg_salary_usd) OVER (ORDER BY period) * 100, 1
    ) as salary_change_percent
FROM covid_periods;

-- 4. Top Emerging Job Titles (2025)
SELECT 
    'Emerging Job Titles in 2025' as analysis_type,
    job_title,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(
        (AVG(salary_in_usd) - (SELECT AVG(salary_in_usd) FROM salaries WHERE work_year = 2025)) / 
        (SELECT AVG(salary_in_usd) FROM salaries WHERE work_year = 2025) * 100, 1
    ) as premium_vs_2025_avg_percent
FROM salaries
WHERE work_year = 2025
GROUP BY job_title
HAVING COUNT(*) >= 10  -- Minimum 10 records for reliability
ORDER BY avg_salary_usd DESC
LIMIT 15;

-- 5. Salary Prediction Insights (for ML modeling)
SELECT 
    'Salary Prediction Factors' as analysis_type,
    'Experience Level' as factor,
    experience_level,
    CASE 
        WHEN experience_level = 'EN' THEN 'Entry-level'
        WHEN experience_level = 'MI' THEN 'Mid-level'
        WHEN experience_level = 'SE' THEN 'Senior-level'
        WHEN experience_level = 'EX' THEN 'Executive'
    END as factor_description,
    COUNT(*) as sample_size,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(STDDEV(salary_in_usd), 0) as salary_std_dev
FROM salaries
GROUP BY experience_level
UNION ALL
SELECT 
    'Salary Prediction Factors',
    'Remote Work',
    CAST(remote_ratio AS VARCHAR),
    CASE 
        WHEN remote_ratio = 0 THEN 'On-site'
        WHEN remote_ratio = 50 THEN 'Hybrid'
        WHEN remote_ratio = 100 THEN 'Fully Remote'
    END,
    COUNT(*),
    ROUND(AVG(salary_in_usd), 0),
    ROUND(STDDEV(salary_in_usd), 0)
FROM salaries
GROUP BY remote_ratio
UNION ALL
SELECT 
    'Salary Prediction Factors',
    'Company Size',
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small'
        WHEN company_size = 'M' THEN 'Medium'
        WHEN company_size = 'L' THEN 'Large'
    END,
    COUNT(*),
    ROUND(AVG(salary_in_usd), 0),
    ROUND(STDDEV(salary_in_usd), 0)
FROM salaries
GROUP BY company_size
ORDER BY factor, factor_description;

-- 6. Business Intelligence Dashboard Summary
SELECT 
    'Business Intelligence Summary' as dashboard_section,
    work_year,
    COUNT(*) as total_records,
    ROUND(AVG(salary_in_usd), 0) as avg_salary_usd,
    ROUND(AVG(CASE WHEN remote_ratio = 100 THEN 1 ELSE 0 END) * 100, 1) as remote_work_percentage,
    ROUND(AVG(CASE WHEN company_size = 'L' THEN 1 ELSE 0 END) * 100, 1) as large_company_percentage,
    ROUND(AVG(CASE WHEN experience_level = 'SE' OR experience_level = 'EX' THEN 1 ELSE 0 END) * 100, 1) as senior_level_percentage
FROM salaries
GROUP BY work_year
ORDER BY work_year;
