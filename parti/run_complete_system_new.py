#!/usr/bin/env python3
"""
Complete System Runner
Runs the entire Environmental SME system with all components
"""

import sys
from pathlib import Path

# Ensure we're working from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'parti'))

import logging
import argparse
import webbrowser
import time
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pymilvus': 'Milvus',
        'sentence_transformers': 'Sentence Transformers',
        'google.generativeai': 'Gemini API',
        'reportlab': 'ReportLab',
        'docx': 'python-docx'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module.split('.')[0])
            logger.info(f"   {name}")
        except ImportError:
            logger.error(f"   {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        logger.error(f"\nMissing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info(" All dependencies installed\n")
    return True


def check_data():
    """Check if required data files exist."""
    logger.info("Checking data files...")
    
    required_paths = [
        project_root / 'partc' / 'milvus_data.db',
        project_root / 'dataset',
        project_root / 'data_json'
    ]
    
    missing = []
    for path in required_paths:
        if path.exists():
            logger.info(f"   {path}")
        else:
            logger.warning(f"  ! {path} - NOT FOUND")
            missing.append(str(path))
    
    if any('milvus_data.db' in str(m) for m in missing):
        logger.error("\n Milvus database not found!")
        logger.error("Run: python partc/partc_run_partc_milvus.py")
        return False
    
    logger.info(" Data files OK\n")
    return True


def check_environment():
    """Check environment variables."""
    logger.info("Checking environment variables...")
    
    import os
    
    env_vars = {
        'GEMINI_API_KEY': 'Gemini API Key'
    }
    
    for var, desc in env_vars.items():
        if os.getenv(var):
            logger.info(f"   {desc}")
        else:
            logger.warning(f"  ! {desc} - NOT SET")
    
    logger.info("")


def run_api_server(host='0.0.0.0', port=8000):
    """Run the API server."""
    logger.info(f"Starting API server on {host}:{port}...")
    
    # Import from the correct location
    import main_api_server
    main_api_server.run_server(host, port)


def open_browser(url, delay=3):
    """Open browser after delay."""
    time.sleep(delay)
    logger.info(f"Opening browser at {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")


def run_complete_system(host='0.0.0.0', port=8000, open_ui=True):
    """Run the complete system."""
    logger.info("="*70)
    logger.info("STARTING COMPLETE ENVIRONMENTAL SME SYSTEM")
    logger.info("="*70)
    
    # Pre-flight checks
    if not check_dependencies():
        return 1
    
    if not check_data():
        return 1
    
    check_environment()
    
    # Open browser in background
    if open_ui:
        ui_url = f"http://localhost:{port}/static/index.html"
        Thread(target=open_browser, args=(ui_url,), daemon=True).start()
    
    # Start API server (blocking)
    logger.info("\n" + "="*70)
    logger.info("SYSTEM READY")
    logger.info("="*70)
    logger.info(f"API Server: http://localhost:{port}")
    logger.info(f"Web UI: http://localhost:{port}/static/index.html")
    logger.info(f"API Docs: http://localhost:{port}/docs")
    logger.info("="*70 + "\n")
    
    try:
        run_api_server(host, port)
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Environmental SME System')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--no-ui', action='store_true', help='Do not open browser')
    
    args = parser.parse_args()
    
    return run_complete_system(args.host, args.port, not args.no_ui)


if __name__ == "__main__":
    sys.exit(main())
