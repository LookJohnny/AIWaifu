"""Quick validation of main.py syntax and imports"""
import sys

try:
    # Test imports
    print("Checking imports...")
    import json
    import time
    print("✓ Standard library imports OK")

    # Try importing our modules (this will fail if dependencies aren't installed)
    try:
        from pydantic import BaseModel, Field
        from fastapi import FastAPI
        print("✓ FastAPI and Pydantic imports OK")
    except ImportError as e:
        print(f"⚠ Dependencies not installed: {e}")
        print("Run: pip install -r requirements.txt")

    # Check syntax by compiling
    print("\nChecking syntax...")
    with open('main.py', 'r') as f:
        compile(f.read(), 'main.py', 'exec')
    print("✓ main.py syntax OK")

    with open('test_client.py', 'r') as f:
        compile(f.read(), 'test_client.py', 'exec')
    print("✓ test_client.py syntax OK")

    print("\n✓ All validation checks passed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run server: python main.py")
    print("3. Test: python test_client.py")

except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
