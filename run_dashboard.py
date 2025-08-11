#!/usr/bin/env python3
"""
Dashboard Launcher Script
This script helps you run the Data Science Salary Analysis Dashboard.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'scikit-learn', 'xgboost', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Data Science Salary Analysis Dashboard...")
    print("📊 The dashboard will open in your default web browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "08_dashboard_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    print("=" * 60)
    print("📊 Data Science Salary Analysis Dashboard Launcher")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists('salaries.csv'):
        print("❌ Error: 'salaries.csv' file not found!")
        print("Please make sure the salaries.csv file is in the current directory.")
        return
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install the missing packages? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                return
        else:
            print("❌ Cannot run dashboard without required packages.")
            return
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
