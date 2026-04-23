"""
Root conftest.py — adds src/ to sys.path so every test module can import
vocalog_ai_api without installing the package.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
