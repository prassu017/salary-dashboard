#!/usr/bin/env python3
"""
Deployment Helper Script for Streamlit Dashboard
This script helps you deploy your dashboard to various platforms.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required_files = [
        '08_dashboard_app.py',
        'salaries.csv',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def test_local():
    """Test if the dashboard runs locally"""
    print("🧪 Testing local deployment...")
    try:
        # Test import
        import streamlit
        import pandas
        import plotly
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def show_deployment_options():
    """Show deployment options"""
    print("\n" + "="*60)
    print("🚀 STREAMLIT DASHBOARD DEPLOYMENT OPTIONS")
    print("="*60)
    
    print("\n1. 🌐 Streamlit Cloud (Recommended - Free)")
    print("   • Go to: https://share.streamlit.io")
    print("   • Sign in with GitHub")
    print("   • Upload your repository")
    print("   • Get a public URL instantly")
    
    print("\n2. 🐳 Docker Deployment")
    print("   • Build: docker build -t salary-dashboard .")
    print("   • Run: docker run -p 8501:8501 salary-dashboard")
    
    print("\n3. 🏠 Local Network Sharing")
    print("   • Run: streamlit run 08_dashboard_app.py --server.address=0.0.0.0")
    print("   • Share your IP address with others")
    
    print("\n4. ☁️ Heroku Deployment")
    print("   • Install Heroku CLI")
    print("   • Run: heroku create your-app-name")
    print("   • Run: git push heroku main")
    
    print("\n5. 📱 Local Testing")
    print("   • Run: streamlit run 08_dashboard_app.py")
    print("   • Access at: http://localhost:8501")

def create_github_repo_guide():
    """Guide for creating GitHub repository"""
    print("\n" + "="*60)
    print("📁 GITHUB REPOSITORY SETUP GUIDE")
    print("="*60)
    
    print("\n1. Create a new repository on GitHub:")
    print("   • Go to: https://github.com/new")
    print("   • Choose repository name (e.g., 'salary-dashboard')")
    print("   • Make it public or private")
    print("   • Don't initialize with README (we already have one)")
    
    print("\n2. Upload your files:")
    print("   • Click 'uploading an existing file'")
    print("   • Drag and drop all your project files")
    print("   • Include: 08_dashboard_app.py, salaries.csv, requirements.txt, etc.")
    print("   • Commit the changes")
    
    print("\n3. Your repository structure should look like:")
    print("   your-repo/")
    print("   ├── 08_dashboard_app.py")
    print("   ├── salaries.csv")
    print("   ├── requirements.txt")
    print("   ├── README.md")
    print("   ├── .gitignore")
    print("   └── .streamlit/")
    print("       └── config.toml")

def open_streamlit_cloud():
    """Open Streamlit Cloud in browser"""
    print("\n🌐 Opening Streamlit Cloud...")
    webbrowser.open("https://share.streamlit.io")
    print("✅ Streamlit Cloud opened in your browser")
    print("📝 Follow the deployment guide in DEPLOYMENT_GUIDE.md")

def main():
    """Main deployment helper"""
    print("🚀 Streamlit Dashboard Deployment Helper")
    print("="*50)
    
    # Check files
    missing_files = check_files()
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the current directory.")
        return
    
    print("✅ All required files found")
    
    # Test local deployment
    if not test_local():
        print("❌ Local test failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Show options
    show_deployment_options()
    
    # Ask user preference
    print("\n" + "-"*60)
    choice = input("\nWhich deployment option would you like to use? (1-5): ").strip()
    
    if choice == "1":
        print("\n🎯 Streamlit Cloud Deployment Selected")
        create_github_repo_guide()
        open_streamlit_cloud()
        
    elif choice == "2":
        print("\n🐳 Docker Deployment Selected")
        print("Run these commands:")
        print("docker build -t salary-dashboard .")
        print("docker run -p 8501:8501 salary-dashboard")
        
    elif choice == "3":
        print("\n🏠 Local Network Sharing Selected")
        print("Run this command:")
        print("streamlit run 08_dashboard_app.py --server.address=0.0.0.0 --server.port=8501")
        
    elif choice == "4":
        print("\n☁️ Heroku Deployment Selected")
        print("Follow the Heroku guide in DEPLOYMENT_GUIDE.md")
        
    elif choice == "5":
        print("\n📱 Local Testing Selected")
        print("Run this command:")
        print("streamlit run 08_dashboard_app.py")
        
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    print("\n" + "="*60)
    print("📚 For detailed instructions, see: DEPLOYMENT_GUIDE.md")
    print("🎯 For presentation integration tips, see the guide")
    print("="*60)

if __name__ == "__main__":
    main()
