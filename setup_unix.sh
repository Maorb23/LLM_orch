#!/bin/bash
# Enhanced Query Classification - Unix/Linux/macOS Setup Script
# Run this to install all required packages and NLTK data

echo "============================================================"
echo "Enhanced Query Classification System - Setup"
echo "============================================================"
echo

echo "Installing required Python packages..."
pip install nltk sentence-transformers scikit-learn numpy

echo
echo "Setting up NLTK data..."
python3 setup_nltk.py

echo
echo "Testing installation..."
python3 -c "
try:
    import nltk
    from nltk.tokenize import word_tokenize
    print('✓ NLTK installation successful')
    
    import sentence_transformers
    print('✓ Sentence Transformers available')
    
    import sklearn
    print('✓ Scikit-learn available')
    
    print('✓ All packages installed successfully!')
    print('You can now run the enhanced classification system.')
    
except ImportError as e:
    print(f'✗ Import error: {e}')
    print('Please check the installation messages above.')
"

echo
echo "============================================================"
echo "Setup completed! Check messages above for any errors."
echo "============================================================"
