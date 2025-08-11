-- BUS AN 512 Data Management and SQL - Group Project
-- Data Science Salary Analysis (2020-2025)
-- File 1: Data Exploration and Basic Statistics

-- 1. Basic Dataset Overview
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT work_year) as years_covered,
    MIN(work_year) as earliest_year,
    MAX(work_year) as latest_year,
    COUNT(DISTINCT job_title) as unique_job_titles,
    COUNT(DISTINCT employee_residence) as countries_represented,
    COUNT(DISTINCT company_location) as company_locations
FROM salaries;

-- 2. Salary Distribution by Year
SELECT 
    work_year,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
    ROUND(MAX(salary_in_usd), 2) as max_salary_usd,
    ROUND(STDDEV(salary_in_usd), 2) as salary_std_dev
FROM salaries
GROUP BY work_year
ORDER BY work_year;

-- 3. Experience Level Analysis
SELECT 
    experience_level,
    CASE 
        WHEN experience_level = 'EN' THEN 'Entry-level / Junior'
        WHEN experience_level = 'MI' THEN 'Mid-level / Intermediate'
        WHEN experience_level = 'SE' THEN 'Senior-level'
        WHEN experience_level = 'EX' THEN 'Executive / Director'
    END as experience_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
    ROUND(MAX(salary_in_usd), 2) as max_salary_usd
FROM salaries
GROUP BY experience_level
ORDER BY avg_salary_usd DESC;

-- 4. Employment Type Distribution
SELECT 
    employment_type,
    CASE 
        WHEN employment_type = 'FT' THEN 'Full-time'
        WHEN employment_type = 'PT' THEN 'Part-time'
        WHEN employment_type = 'CT' THEN 'Contract'
        WHEN employment_type = 'FL' THEN 'Freelance'
    END as employment_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM salaries), 2) as percentage
FROM salaries
GROUP BY employment_type
ORDER BY record_count DESC;

-- 5. Company Size Analysis
SELECT 
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50 employees)'
        WHEN company_size = 'M' THEN 'Medium (51-500 employees)'
        WHEN company_size = 'L' THEN 'Large (501+ employees)'
    END as company_size_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM salaries), 2) as percentage
FROM salaries
GROUP BY company_size
ORDER BY avg_salary_usd DESC;

-- 6. Remote Work Analysis
SELECT 
    remote_ratio,
    CASE 
        WHEN remote_ratio = 0 THEN 'On-site (0%)'
        WHEN remote_ratio = 50 THEN 'Hybrid (50%)'
        WHEN remote_ratio = 100 THEN 'Fully Remote (100%)'
    END as remote_work_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM salaries), 2) as percentage
FROM salaries
GROUP BY remote_ratio
ORDER BY remote_ratio;

-- 7. Top 10 Job Titles by Average Salary
SELECT 
    job_title,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
    ROUND(MAX(salary_in_usd), 2) as max_salary_usd
FROM salaries
GROUP BY job_title
HAVING COUNT(*) >= 10  -- Only include job titles with at least 10 records
ORDER BY avg_salary_usd DESC
LIMIT 10;

-- 8. Top 10 Countries by Average Salary
SELECT 
    employee_residence,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
    ROUND(MAX(salary_in_usd), 2) as max_salary_usd
FROM salaries
GROUP BY employee_residence
HAVING COUNT(*) >= 50  -- Only include countries with at least 50 records
ORDER BY avg_salary_usd DESC
LIMIT 10;
