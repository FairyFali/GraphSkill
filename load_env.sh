#!/bin/bash
# Load environment variables from .env file
# Usage: source load_env.sh

if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment variables loaded:"
    echo "  - HF_TOKEN: ${HF_TOKEN:0:10}..."
    echo "  - TOGETHER_API_KEY: ${TOGETHER_API_KEY:0:10}..."
    echo "  - DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:0:10}..."
    echo "  - OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
else
    echo "✗ Error: .env file not found!"
    echo "Please create .env file from .env.example template"
    return 1
fi
