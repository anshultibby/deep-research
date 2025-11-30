#!/bin/bash

echo "⚛️  Starting Frontend (Next.js)..."
echo ""

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd frontend
    npm install
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    cd ..
else
    echo "✅ Dependencies OK"
fi

# Check if backend is running
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Warning: Backend doesn't seem to be running!"
    echo "   Make sure to start the backend first in another terminal:"
    echo "   ./run-backend.sh"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to cancel..."
fi

echo ""

# Start frontend
cd frontend

echo "📦 Starting Next.js on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run dev

