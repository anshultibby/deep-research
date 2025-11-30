#!/bin/bash

echo "🐍 Starting Backend (FastAPI)..."
echo ""

# Change to backend directory first
cd backend

# Find Python 3.11+ command
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo "✅ Found Python $VERSION"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.11+ not found. Please install Python 3.11 or 3.12."
    echo "   Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check if .env exists in backend/, create from template if not
if [ ! -f .env ]; then
    echo "⚠️  No .env file found in backend/. Creating from template..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Created backend/.env file"
        echo ""
        echo "❗ IMPORTANT: Add your API keys to backend/.env:"
        echo "   - OPENAI_API_KEY=sk-..."
        echo "   - SERPER_API_KEY=..."
        echo ""
        read -p "Press Enter after adding your API keys to continue..."
    else
        echo "❌ env.example not found in backend/!"
        exit 1
    fi
fi

# Create virtual environment in backend/ if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment in backend/venv..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
if [ $? -eq 0 ]; then
    echo "✅ Dependencies ready"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""

# Start backend
echo "📦 Starting FastAPI server on http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python api.py

