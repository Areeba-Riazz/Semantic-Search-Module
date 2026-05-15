"""
Main entry point for the AI Research Assistant — Phase 1: Semantic Search Module.

This module provides a CLI entry point that launches the Streamlit GUI.
It can also be used to verify that all dependencies are properly installed.

Usage:
    streamlit run app/gui.py
    -- or --
    python app/main.py   (this will launch Streamlit automatically)
"""

import sys
import os
import subprocess

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_dependencies():
    """Quick smoke-test that critical packages can be imported."""
    required = [
        "streamlit",
        "langchain",
        "langchain_huggingface",
        "langchain_community",
        "sentence_transformers",
        "faiss",
        "chromadb",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("❌ Missing packages:")
        for m in missing:
            print(f"   - {m}")
        print("\nRun:  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies installed.")


def main():
    """Launch the Streamlit GUI."""
    check_dependencies()

    gui_path = os.path.join(os.path.dirname(__file__), "gui.py")
    print(f"🚀 Launching Streamlit GUI …  ({gui_path})")
    subprocess.run([sys.executable, "-m", "streamlit", "run", gui_path], check=True)


if __name__ == "__main__":
    main()
