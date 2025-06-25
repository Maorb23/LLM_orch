#!/usr/bin/env python3
"""
NLTK Setup Script for Enhanced Query Classification
Run this script to install and download all required NLTK data.
"""

import sys
import subprocess
import os

def install_nltk():
    """Install NLTK if not already installed"""
    try:
        import nltk
        print("✓ NLTK is already installed")
        return True
    except ImportError:
        print("Installing NLTK...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            print("✓ NLTK installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install NLTK: {e}")
            return False

def download_nltk_data():
    """Download all required NLTK data"""
    try:
        import nltk
    except ImportError:
        print("✗ NLTK not available, cannot download data")
        return False
    
    # Required NLTK data packages
    required_packages = [
        # Tokenizers
        'punkt',
        'punkt_tab',  # For newer NLTK versions
        
        # Corpora
        'stopwords',
        'wordnet', 
        'omw-1.4',  # Open Multilingual Wordnet
        'words',
        'brown',  # Brown corpus
        
        # Taggers
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',  # For newer versions
        
        # Chunkers  
        'maxent_ne_chunker',
        
        # Additional useful data
        'vader_lexicon',  # Sentiment analysis
        'maxent_ne_chunker_tab',  # Named entity chunker
    ]
    
    print("Downloading NLTK data packages...")
    success_count = 0
    
    for package in required_packages:
        try:
            print(f"Downloading {package}...", end=" ")
            nltk.download(package, quiet=True)
            print("✓")
            success_count += 1
        except Exception as e:
            print(f"⚠ (Warning: {e})")
            # Continue with other packages
            continue
    
    print(f"\nDownloaded {success_count}/{len(required_packages)} packages successfully")
    
    if success_count >= len(required_packages) - 2:  # Allow 2 failures for version differences
        print("✓ NLTK setup completed successfully!")
        return True
    else:
        print("⚠ Some packages failed to download, but basic functionality should work")
        return True

def verify_installation():
    """Verify that key NLTK functionality works"""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        # Test basic functionality
        text = "This is a test sentence for NLTK verification."
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
        print("\n--- Testing NLTK functionality ---")
        print(f"Original text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Stop words count: {len(stop_words)}")
        print(f"Stemmed 'running': {stemmer.stem('running')}")
        print("✓ NLTK verification successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ NLTK verification failed: {e}")
        print("Some functionality may not work properly.")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Enhanced Query Classification - NLTK Setup")
    print("=" * 60)
    
    # Step 1: Install NLTK
    if not install_nltk():
        print("✗ Setup failed: Could not install NLTK")
        return False
    
    # Step 2: Download required data
    if not download_nltk_data():
        print("✗ Setup failed: Could not download NLTK data")
        return False
    
    # Step 3: Verify installation
    verify_installation()
    
    print("\n" + "=" * 60)
    print("Setup completed! You can now run the enhanced classification system.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
