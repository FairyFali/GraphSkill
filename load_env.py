"""
Load environment variables from .env file using python-dotenv.

This script automatically loads API keys from .env file when imported
or run directly. All runner scripts should import this at the beginning.

Usage:
    # In your Python scripts
    from load_env import load_api_keys
    load_api_keys()

    # Or simply import to auto-load
    import load_env

    # Or run directly
    python load_env.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_api_keys(verbose=True):
    """
    Load API keys from .env file.

    Args:
        verbose (bool): If True, print status messages

    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    env_path = Path(__file__).parent / '.env'

    if not env_path.exists():
        if verbose:
            print("⚠️  Warning: .env file not found!")
            print("   Please create .env file from .env.example template")
            print(f"   Expected location: {env_path}")
        return False

    # Load environment variables
    load_dotenv(env_path, override=True)

    # Check which keys are loaded
    keys = {
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'TOGETHER_API_KEY': os.getenv('TOGETHER_API_KEY'),
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    }

    if verbose:
        print("✓ Environment variables loaded from .env file:")
        for key, value in keys.items():
            if value:
                # Show first 10 chars for verification
                masked = f"{value[:10]}..." if len(value) > 10 else value[:5] + "..."
                print(f"  - {key}: {masked}")
            else:
                print(f"  - {key}: ✗ Not set")

    return True


# Auto-load when imported
load_api_keys(verbose=False)


if __name__ == "__main__":
    # Run with verbose output when executed directly
    print("=" * 50)
    print("API Key Loader")
    print("=" * 50)

    success = load_api_keys(verbose=True)

    if success:
        print("\n✓ All API keys loaded successfully!")
        print("\nYou can now run your experiments:")
        print("  python runners/graphtutor/run_graphtutor_tasks_code_only_baselines.py")
    else:
        print("\n✗ Failed to load API keys")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env")
        print("  2. Fill in your actual API keys")
        print("  3. Run this script again to verify")
