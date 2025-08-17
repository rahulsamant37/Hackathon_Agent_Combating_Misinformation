#!/usr/bin/env python3
"""
Main entry point for the Misinformation Detection Tool.

This script provides a unified interface to start either the FastAPI backend
or the Streamlit frontend application.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def start_api():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✋ FastAPI server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting FastAPI server: {e}")
        sys.exit(1)


def start_streamlit():
    """Start the Streamlit frontend application."""
    print("🎨 Starting Streamlit frontend application...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✋ Streamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit app: {e}")
        sys.exit(1)


def start_both():
    """Start both FastAPI and Streamlit in parallel."""
    import threading
    import time
    
    print("🚀 Starting both FastAPI backend and Streamlit frontend...")
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Give FastAPI time to start
    time.sleep(3)
    
    # Start Streamlit in main thread
    start_streamlit()


def check_environment():
    """Check if the environment is properly configured."""
    env_file = Path(".env")
    config_file = Path("config/config.yaml")
    
    issues = []
    
    if not env_file.exists():
        issues.append("❌ .env file not found. Copy .env.example to .env and configure your API keys.")
    
    if not config_file.exists():
        issues.append("❌ config/config.yaml not found. Please ensure configuration files are in place.")
    
    if issues:
        print("⚠️  Environment Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Please resolve these issues before starting the application.")
        return False
    
    print("✅ Environment check passed!")
    return True


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Misinformation Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py api          # Start only FastAPI backend
  python main.py streamlit    # Start only Streamlit frontend  
  python main.py both         # Start both services
  python main.py check        # Check environment configuration
        """
    )
    
    parser.add_argument(
        "command",
        choices=["api", "streamlit", "both", "check"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment configuration check"
    )
    
    args = parser.parse_args()
    
    print("🔍 AI-Powered Misinformation Detection Tool")
    print("=" * 50)
    
    # Check environment unless explicitly skipped
    if not args.skip_check and args.command != "check":
        if not check_environment():
            sys.exit(1)
    
    # Execute the requested command
    if args.command == "api":
        start_api()
    elif args.command == "streamlit":
        start_streamlit()
    elif args.command == "both":
        start_both()
    elif args.command == "check":
        check_environment()
    
    print("\n👋 Thank you for using the Misinformation Detection Tool!")


if __name__ == "__main__":
    main()