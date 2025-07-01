#!/bin/bash

# Setup Conda Environment for Speech-to-Speech Translation
# Creates a conda environment with Python 3.11.13 and installs all dependencies

echo "🔧 CONDA ENVIRONMENT SETUP"
echo "=========================="
echo "Creating conda environment: speech-to-speech"
echo "Python version: 3.11.13"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "🐍 Creating conda environment..."
conda create -n speech-to-speech python=3.11.13 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

echo "✅ Conda environment created successfully!"
echo ""

# Instructions for activation and next steps
echo "🎯 NEXT STEPS:"
echo "=============="
echo "1. Activate the environment:"
echo "   conda activate speech-to-speech"
echo ""
echo "2. Install dependencies:"
echo "   ./install_deps.sh"
echo ""
echo "3. Initialize models:"
echo "   python orchestrator.py"
echo ""
echo "🔄 To activate the environment in future sessions:"
echo "   conda activate speech-to-speech"
echo ""
echo "🗑️  To remove the environment (if needed):"
echo "   conda remove -n speech-to-speech --all" 