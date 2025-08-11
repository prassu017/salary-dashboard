-- BUS AN 512 Data Management and SQL - Group Project
-- Data Science Salary Analysis (2020-2025)
-- File 3: Advanced Analytics and Complex Insights

-- 1. Salary Premium Analysis: Remote vs On-site by Experience Level
WITH remote_premium AS (
    SELECT 
        experience_level,
        CASE 
            WHEN experience_level = 'EN' THEN 'Entry-level / Junior'
            WHEN experience_level = 'MI' THEN 'Mid-level / Intermediate'
            WHEN experience_level = 'SE' THEN 'Senior-level'
            WHEN experience_level = 'EX' THEN 'Executive / Director'
        END as experience_description,
        remote_ratio,
        CASE 
            WHEN remote_ratio = 0 THEN 'On-site'
            WHEN remote_ratio = 50 THEN 'Hybrid'
            WHEN remote_ratio = 100 THEN 'Fully Remote'
        END as work_setting,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd
    FROM salaries
    GROUP BY experience_level, remote_ratio
    HAVING COUNT(*) >= 20  -- Minimum 20 records for reliable comparison
)
SELECT 
    experience_level,
    experience_description,
    work_setting,
    record_count,
    avg_salary_usd,
    ROUND(
        (avg_salary_usd - LAG(avg_salary_usd) OVER (PARTITION BY experience_level ORDER BY remote_ratio)) / 
        LAG(avg_salary_usd) OVER (PARTITION BY experience_level ORDER BY remote_ratio) * 100, 2
    ) as premium_vs_onsite_percent
FROM remote_premium
ORDER BY experience_level, remote_ratio;

-- 2. Company Size vs Experience Level Salary Analysis
SELECT 
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50 employees)'
        WHEN company_size = 'M' THEN 'Medium (51-500 employees)'
        WHEN company_size = 'L' THEN 'Large (501+ employees)'
    END as company_size_description,
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
GROUP BY company_size, experience_level
HAVING COUNT(*) >= 10  -- Minimum 10 records for reliable analysis
ORDER BY company_size, experience_level;

-- 3. Geographic Salary Disparity Analysis
WITH country_stats AS (
    SELECT 
        employee_residence,
        COUNT(*) as total_records,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
        ROUND(STDDEV(salary_in_usd), 2) as salary_std_dev,
        ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
        ROUND(MAX(salary_in_usd), 2) as max_salary_usd
    FROM salaries
    GROUP BY employee_residence
    HAVING COUNT(*) >= 50  -- Minimum 50 records per country
),
global_avg AS (
    SELECT ROUND(AVG(salary_in_usd), 2) as global_avg_salary
    FROM salaries
)
SELECT 
    c.employee_residence,
    c.total_records,
    c.avg_salary_usd,
    c.salary_std_dev,
    c.min_salary_usd,
    c.max_salary_usd,
    g.global_avg_salary,
    ROUND((c.avg_salary_usd - g.global_avg_salary) / g.global_avg_salary * 100, 2) as deviation_from_global_avg_percent,
    ROUND(c.salary_std_dev / c.avg_salary_usd * 100, 2) as coefficient_of_variation_percent
FROM country_stats c
CROSS JOIN global_avg g
ORDER BY c.avg_salary_usd DESC;

-- 4. Job Title Salary Distribution Analysis
WITH job_title_stats AS (
    SELECT 
        job_title,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
        ROUND(STDDEV(salary_in_usd), 2) as salary_std_dev,
        ROUND(MIN(salary_in_usd), 2) as min_salary_usd,
        ROUND(MAX(salary_in_usd), 2) as max_salary_usd,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary_in_usd), 2) as q1_salary,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary_in_usd), 2) as q3_salary
    FROM salaries
    GROUP BY job_title
    HAVING COUNT(*) >= 20  -- Minimum 20 records per job title
)
SELECT 
    job_title,
    record_count,
    avg_salary_usd,
    salary_std_dev,
    min_salary_usd,
    max_salary_usd,
    q1_salary,
    q3_salary,
    ROUND(salary_std_dev / avg_salary_usd * 100, 2) as coefficient_of_variation_percent,
    ROUND((q3_salary - q1_salary) / avg_salary_usd * 100, 2) as iqr_as_percent_of_mean
FROM job_title_stats
ORDER BY avg_salary_usd DESC
LIMIT 20;

-- 5. Employment Type vs Company Size Analysis
SELECT 
    employment_type,
    CASE 
        WHEN employment_type = 'FT' THEN 'Full-time'
        WHEN employment_type = 'PT' THEN 'Part-time'
        WHEN employment_type = 'CT' THEN 'Contract'
        WHEN employment_type = 'FL' THEN 'Freelance'
    END as employment_description,
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50 employees)'
        WHEN company_size = 'M' THEN 'Medium (51-500 employees)'
        WHEN company_size = 'L' THEN 'Large (501+ employees)'
    END as company_size_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY employment_type), 2) as percentage_of_employment_type
FROM salaries
GROUP BY employment_type, company_size
HAVING COUNT(*) >= 10  -- Minimum 10 records for reliable analysis
ORDER BY employment_type, company_size;

-- 6. Salary Growth Rate Analysis by Job Title (2020-2025)
WITH job_title_growth AS (
    SELECT 
        job_title,
        work_year,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd
    FROM salaries
    WHERE work_year IN (2020, 2025)
    GROUP BY job_title, work_year
    HAVING COUNT(*) >= 5  -- Minimum 5 records per year
),
growth_calculation AS (
    SELECT 
        j1.job_title,
        j1.record_count as records_2020,
        j1.avg_salary_usd as avg_salary_2020,
        j5.record_count as records_2025,
        j5.avg_salary_usd as avg_salary_2025,
        ROUND(
            (j5.avg_salary_usd - j1.avg_salary_usd) / j1.avg_salary_usd * 100, 2
        ) as salary_growth_percent,
        ROUND(
            POWER(j5.avg_salary_usd / j1.avg_salary_usd, 1.0/5) - 1, 4
        ) * 100 as compound_annual_growth_rate_percent
    FROM job_title_growth j1
    JOIN job_title_growth j5 ON j1.job_title = j5.job_title
    WHERE j1.work_year = 2020 AND j5.work_year = 2025
)
SELECT 
    job_title,
    records_2020,
    avg_salary_2020,
    records_2025,
    avg_salary_2025,
    salary_growth_percent,
    compound_annual_growth_rate_percent
FROM growth_calculation
ORDER BY compound_annual_growth_rate_percent DESC
LIMIT 20;

-- 7. Remote Work Adoption by Country
WITH country_remote_stats AS (
    SELECT 
        employee_residence,
        remote_ratio,
        CASE 
            WHEN remote_ratio = 0 THEN 'On-site'
            WHEN remote_ratio = 50 THEN 'Hybrid'
            WHEN remote_ratio = 100 THEN 'Fully Remote'
        END as work_setting,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd
    FROM salaries
    GROUP BY employee_residence, remote_ratio
    HAVING COUNT(*) >= 10  -- Minimum 10 records per country per setting
),
country_totals AS (
    SELECT 
        employee_residence,
        SUM(record_count) as total_records
    FROM country_remote_stats
    GROUP BY employee_residence
    HAVING SUM(record_count) >= 100  -- Minimum 100 total records per country
)
SELECT 
    c.employee_residence,
    c.work_setting,
    c.record_count,
    c.avg_salary_usd,
    ROUND(c.record_count * 100.0 / t.total_records, 2) as percentage_of_country
FROM country_remote_stats c
JOIN country_totals t ON c.employee_residence = t.employee_residence
ORDER BY c.employee_residence, c.remote_ratio;

-- 8. Experience Level Distribution by Company Size and Year
SELECT 
    work_year,
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50 employees)'
        WHEN company_size = 'M' THEN 'Medium (51-500 employees)'
        WHEN company_size = 'L' THEN 'Large (501+ employees)'
    END as company_size_description,
    experience_level,
    CASE 
        WHEN experience_level = 'EN' THEN 'Entry-level / Junior'
        WHEN experience_level = 'MI' THEN 'Mid-level / Intermediate'
        WHEN experience_level = 'SE' THEN 'Senior-level'
        WHEN experience_level = 'EX' THEN 'Executive / Director'
    END as experience_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY work_year, company_size), 2) as percentage_of_company_size
FROM salaries
GROUP BY work_year, company_size, experience_level
HAVING COUNT(*) >= 5  -- Minimum 5 records for reliable analysis
ORDER BY work_year, company_size, experience_level;
