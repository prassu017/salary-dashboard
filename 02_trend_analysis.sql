-- BUS AN 512 Data Management and SQL - Group Project
-- Data Science Salary Analysis (2020-2025)
-- File 2: Trend Analysis and Time Series Insights

-- 1. Salary Growth Trends by Year and Experience Level
SELECT 
    work_year,
    experience_level,
    CASE 
        WHEN experience_level = 'EN' THEN 'Entry-level / Junior'
        WHEN experience_level = 'MI' THEN 'Mid-level / Intermediate'
        WHEN experience_level = 'SE' THEN 'Senior-level'
        WHEN experience_level = 'EX' THEN 'Executive / Director'
    END as experience_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(LAG(AVG(salary_in_usd)) OVER (PARTITION BY experience_level ORDER BY work_year), 2) as prev_year_avg,
    ROUND(
        (AVG(salary_in_usd) - LAG(AVG(salary_in_usd)) OVER (PARTITION BY experience_level ORDER BY work_year)) / 
        LAG(AVG(salary_in_usd)) OVER (PARTITION BY experience_level ORDER BY work_year) * 100, 2
    ) as year_over_year_growth_percent
FROM salaries
GROUP BY work_year, experience_level
ORDER BY experience_level, work_year;

-- 2. Remote Work Adoption Trends Over Time
SELECT 
    work_year,
    remote_ratio,
    CASE 
        WHEN remote_ratio = 0 THEN 'On-site (0%)'
        WHEN remote_ratio = 50 THEN 'Hybrid (50%)'
        WHEN remote_ratio = 100 THEN 'Fully Remote (100%)'
    END as remote_work_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY work_year), 2) as percentage_of_year
FROM salaries
GROUP BY work_year, remote_ratio
ORDER BY work_year, remote_ratio;

-- 3. Company Size Distribution Trends
SELECT 
    work_year,
    company_size,
    CASE 
        WHEN company_size = 'S' THEN 'Small (1-50 employees)'
        WHEN company_size = 'M' THEN 'Medium (51-500 employees)'
        WHEN company_size = 'L' THEN 'Large (501+ employees)'
    END as company_size_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY work_year), 2) as percentage_of_year
FROM salaries
GROUP BY work_year, company_size
ORDER BY work_year, company_size;

-- 4. Top Job Titles Evolution (2020 vs 2025)
WITH job_title_evolution AS (
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
job_comparison AS (
    SELECT 
        j1.job_title,
        j1.record_count as records_2020,
        j1.avg_salary_usd as avg_salary_2020,
        j5.record_count as records_2025,
        j5.avg_salary_usd as avg_salary_2025,
        ROUND(
            (j5.avg_salary_usd - j1.avg_salary_usd) / j1.avg_salary_usd * 100, 2
        ) as salary_growth_percent
    FROM job_title_evolution j1
    JOIN job_title_evolution j5 ON j1.job_title = j5.job_title
    WHERE j1.work_year = 2020 AND j5.work_year = 2025
)
SELECT 
    job_title,
    records_2020,
    avg_salary_2020,
    records_2025,
    avg_salary_2025,
    salary_growth_percent
FROM job_comparison
ORDER BY salary_growth_percent DESC
LIMIT 15;

-- 5. Geographic Salary Trends - Top Countries
WITH country_yearly_stats AS (
    SELECT 
        employee_residence,
        work_year,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd
    FROM salaries
    GROUP BY employee_residence, work_year
    HAVING COUNT(*) >= 10  -- Minimum 10 records per year per country
),
top_countries AS (
    SELECT DISTINCT employee_residence
    FROM country_yearly_stats
    WHERE work_year = 2025
    ORDER BY avg_salary_usd DESC
    LIMIT 10
)
SELECT 
    c.employee_residence,
    c.work_year,
    c.record_count,
    c.avg_salary_usd,
    ROUND(LAG(c.avg_salary_usd) OVER (PARTITION BY c.employee_residence ORDER BY c.work_year), 2) as prev_year_avg,
    ROUND(
        (c.avg_salary_usd - LAG(c.avg_salary_usd) OVER (PARTITION BY c.employee_residence ORDER BY c.work_year)) / 
        LAG(c.avg_salary_usd) OVER (PARTITION BY c.employee_residence ORDER BY c.work_year) * 100, 2
    ) as year_over_year_growth_percent
FROM country_yearly_stats c
JOIN top_countries t ON c.employee_residence = t.employee_residence
ORDER BY c.employee_residence, c.work_year;

-- 6. Employment Type Trends Over Time
SELECT 
    work_year,
    employment_type,
    CASE 
        WHEN employment_type = 'FT' THEN 'Full-time'
        WHEN employment_type = 'PT' THEN 'Part-time'
        WHEN employment_type = 'CT' THEN 'Contract'
        WHEN employment_type = 'FL' THEN 'Freelance'
    END as employment_description,
    COUNT(*) as record_count,
    ROUND(AVG(salary_in_usd), 2) as avg_salary_usd,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY work_year), 2) as percentage_of_year
FROM salaries
GROUP BY work_year, employment_type
ORDER BY work_year, employment_type;

-- 7. Salary Range Analysis by Year
SELECT 
    work_year,
    COUNT(*) as total_records,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary_in_usd), 2) as q1_salary,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary_in_usd), 2) as median_salary,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary_in_usd), 2) as q3_salary,
    ROUND(AVG(salary_in_usd), 2) as mean_salary,
    ROUND(STDDEV(salary_in_usd), 2) as salary_std_dev
FROM salaries
GROUP BY work_year
ORDER BY work_year;

-- 8. Experience Level Salary Growth by Year
WITH experience_yearly AS (
    SELECT 
        work_year,
        experience_level,
        CASE 
            WHEN experience_level = 'EN' THEN 'Entry-level / Junior'
            WHEN experience_level = 'MI' THEN 'Mid-level / Intermediate'
            WHEN experience_level = 'SE' THEN 'Senior-level'
            WHEN experience_level = 'EX' THEN 'Executive / Director'
        END as experience_description,
        COUNT(*) as record_count,
        ROUND(AVG(salary_in_usd), 2) as avg_salary_usd
    FROM salaries
    GROUP BY work_year, experience_level
)
SELECT 
    work_year,
    experience_level,
    experience_description,
    record_count,
    avg_salary_usd,
    ROUND(LAG(avg_salary_usd) OVER (PARTITION BY experience_level ORDER BY work_year), 2) as prev_year_avg,
    ROUND(
        (avg_salary_usd - LAG(avg_salary_usd) OVER (PARTITION BY experience_level ORDER BY work_year)) / 
        LAG(avg_salary_usd) OVER (PARTITION BY experience_level ORDER BY work_year) * 100, 2
    ) as year_over_year_growth_percent
FROM experience_yearly
ORDER BY experience_level, work_year;
