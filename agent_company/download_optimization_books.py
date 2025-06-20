#!/usr/bin/env python3
"""
Download optimization-relevant mathematical books from the repository
Focuses on books that will help optimize neural network training
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse
import re

# Base URL for the repository
BASE_URL = "https://raw.githubusercontent.com/"

# Most optimization-relevant books from the list
OPTIMIZATION_BOOKS = [
    {
        "title": "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
        "filename": "algorithms_for_optimization_mit.pdf",
        "url": "https://raw.githubusercontent.com/optimization-book/optimization-book/master/optimization-book.pdf"
    },
    {
        "title": "Theory of matrices, Volume I by Gantmacher",
        "filename": "theory_of_matrices_vol1_gantmacher.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Theory%20of%20matrices%20Volume%20I%20by%20Gantmacher.pdf"
    },
    {
        "title": "Theory of matrices, Volume II by Gantmacher", 
        "filename": "theory_of_matrices_vol2_gantmacher.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Theory%20of%20matrices%20Volume%20II%20by%20Gantmacher.pdf"
    },
    {
        "title": "A Course Of Mathematical Analysis by A. Ya. Khinchin",
        "filename": "course_mathematical_analysis_khinchin.pdf", 
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/A%20Course%20Of%20Mathematical%20Analysis%20by%20A.%20Ya.%20Khinchin.pdf"
    },
    {
        "title": "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
        "filename": "course_mathematical_analysis_nikolsky.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/A%20Course%20Of%20Mathematical%20Analysis%20Vol%201%20by%20S.%20M.%20Nikolsky.pdf"
    },
    {
        "title": "Linear Algebra by V. V. Voyevodin",
        "filename": "linear_algebra_voyevodin.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Linear%20Algebra%20by%20V.%20V.%20Voyevodin.pdf"
    },
    {
        "title": "Problems In Linear Algebra by I. V. Proskuryakov",
        "filename": "problems_linear_algebra_proskuryakov.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Problems%20In%20Linear%20Algebra%20by%20I.%20V.%20Proskuryakov.pdf"
    },
    {
        "title": "Calculus Of Variations by I.M. Gelfand; S.V. Fomin",
        "filename": "calculus_variations_gelfand_fomin.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Calculus%20Of%20Variations%20by%20I.M.%20Gelfand;%20S.V.%20Fomin.pdf"
    },
    {
        "title": "Optimal control by Alekseev, Tikhomirov and Fomin",
        "filename": "optimal_control_alekseev.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Optimal%20control%20by%20Alekseev,%20Tikhomirov%20and%20Fomin.pdf"
    },
    {
        "title": "Probability - First Steps by E.S. Wentzel",
        "filename": "probability_first_steps_wentzel.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Probability%20-%20First%20Steps%20by%20E.S.%20Wentzel.pdf"
    },
    {
        "title": "Operations research by Elena Wentzel",
        "filename": "operations_research_wentzel.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/Operations%20research%20by%20Elena%20Wentzel.pdf"
    },
    {
        "title": "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
        "filename": "mathematics_machine_learning_deisenroth.pdf",
        "url": "https://raw.githubusercontent.com/math-books/math-books/main/MATHEMATICS%20FOR%20MACHINE%20LEARNING%20by%20Marc%20Peter%20Deisenroth.pdf"
    }
]

def download_file(url, filename, download_dir="optimization_books"):
    """Download a file with progress tracking and error handling"""
    try:
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        filepath = os.path.join(download_dir, filename)
        
        print(f"Downloading: {filename}")
        print(f"URL: {url}")
        
        # Download with timeout and retry logic
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Get file size for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                
                print(f"\n✅ Successfully downloaded: {filename}")
                print(f"   Size: {os.path.getsize(filepath)} bytes")
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Failed to download {filename} after 3 attempts")
                    return False
                    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def analyze_optimization_relevance():
    """Analyze which books are most relevant for neural network optimization"""
    print("🔍 Analyzing optimization relevance of available books...")
    print()
    
    # Categorize books by relevance
    categories = {
        "Essential Optimization": [
            "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
            "Optimal control by Alekseev, Tikhomirov and Fomin",
            "Operations research by Elena Wentzel"
        ],
        "Linear Algebra & Matrices": [
            "Theory of matrices, Volume I by Gantmacher",
            "Theory of matrices, Volume II by Gantmacher", 
            "Linear Algebra by V. V. Voyevodin",
            "Problems In Linear Algebra by I. V. Proskuryakov"
        ],
        "Mathematical Analysis": [
            "A Course Of Mathematical Analysis by A. Ya. Khinchin",
            "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
            "Calculus Of Variations by I.M. Gelfand; S.V. Fomin"
        ],
        "Machine Learning Math": [
            "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
            "Probability - First Steps by E.S. Wentzel"
        ]
    }
    
    for category, books in categories.items():
        print(f"📚 {category}:")
        for book in books:
            print(f"   • {book}")
        print()
    
    return categories

def main():
    """Main function to download optimization books"""
    print("🚀 Optimization Book Downloader")
    print("=" * 50)
    
    # Analyze relevance
    categories = analyze_optimization_relevance()
    
    print("📥 Starting downloads...")
    print("=" * 50)
    
    successful_downloads = 0
    total_downloads = len(OPTIMIZATION_BOOKS)
    
    for book in OPTIMIZATION_BOOKS:
        print(f"\n📖 {book['title']}")
        print("-" * 40)
        
        if download_file(book['url'], book['filename']):
            successful_downloads += 1
        
        # Small delay between downloads
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📊 Download Summary:")
    print(f"   Successful: {successful_downloads}/{total_downloads}")
    print(f"   Failed: {total_downloads - successful_downloads}/{total_downloads}")
    
    if successful_downloads > 0:
        print(f"\n📁 Books downloaded to: optimization_books/")
        print("\n🎯 These books will help optimize:")
        print("   • Neural network training algorithms")
        print("   • Gradient descent optimization")
        print("   • Matrix operations and linear algebra")
        print("   • Mathematical analysis for ML")
        print("   • Probability theory for uncertainty")
        print("   • Calculus of variations for optimal control")

if __name__ == "__main__":
    main() 