#!/usr/bin/env python3
"""
Download All Optimization Books from the Repository
Comprehensive downloader for mathematical optimization resources
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse, quote
import re

# Base URLs for different repositories
REPO_URLS = [
    "https://raw.githubusercontent.com/math-books/math-books/main/",
    "https://raw.githubusercontent.com/optimization-book/optimization-book/master/",
    "https://github.com/math-books/math-books/raw/main/",
    "https://github.com/optimization-book/optimization-book/raw/master/"
]

# Comprehensive list of optimization-relevant books
OPTIMIZATION_BOOKS = [
    # Essential Optimization Books
    {
        "title": "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
        "filename": "algorithms_for_optimization_mit.pdf",
        "urls": [
            "https://raw.githubusercontent.com/optimization-book/optimization-book/master/optimization-book.pdf",
            "https://github.com/optimization-book/optimization-book/raw/master/optimization-book.pdf"
        ]
    },
    
    # Matrix Analysis and Linear Algebra
    {
        "title": "Theory of matrices, Volume I by Gantmacher",
        "filename": "theory_of_matrices_vol1_gantmacher.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Theory%20of%20matrices%20Volume%20I%20by%20Gantmacher.pdf",
            "https://github.com/math-books/math-books/raw/main/Theory%20of%20matrices%20Volume%20I%20by%20Gantmacher.pdf"
        ]
    },
    {
        "title": "Theory of matrices, Volume II by Gantmacher",
        "filename": "theory_of_matrices_vol2_gantmacher.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Theory%20of%20matrices%20Volume%20II%20by%20Gantmacher.pdf",
            "https://github.com/math-books/math-books/raw/main/Theory%20of%20matrices%20Volume%20II%20by%20Gantmacher.pdf"
        ]
    },
    {
        "title": "Linear Algebra by V. V. Voyevodin",
        "filename": "linear_algebra_voyevodin.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Linear%20Algebra%20by%20V.%20V.%20Voyevodin.pdf",
            "https://github.com/math-books/math-books/raw/main/Linear%20Algebra%20by%20V.%20V.%20Voyevodin.pdf"
        ]
    },
    {
        "title": "Problems In Linear Algebra by I. V. Proskuryakov",
        "filename": "problems_linear_algebra_proskuryakov.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Problems%20In%20Linear%20Algebra%20by%20I.%20V.%20Proskuryakov.pdf",
            "https://github.com/math-books/math-books/raw/main/Problems%20In%20Linear%20Algebra%20by%20I.%20V.%20Proskuryakov.pdf"
        ]
    },
    
    # Mathematical Analysis
    {
        "title": "A Course Of Mathematical Analysis by A. Ya. Khinchin",
        "filename": "course_mathematical_analysis_khinchin.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/A%20Course%20Of%20Mathematical%20Analysis%20by%20A.%20Ya.%20Khinchin.pdf",
            "https://github.com/math-books/math-books/raw/main/A%20Course%20Of%20Mathematical%20Analysis%20by%20A.%20Ya.%20Khinchin.pdf"
        ]
    },
    {
        "title": "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
        "filename": "course_mathematical_analysis_nikolsky.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/A%20Course%20Of%20Mathematical%20Analysis%20Vol%201%20by%20S.%20M.%20Nikolsky.pdf",
            "https://github.com/math-books/math-books/raw/main/A%20Course%20Of%20Mathematical%20Analysis%20Vol%201%20by%20S.%20M.%20Nikolsky.pdf"
        ]
    },
    {
        "title": "Calculus Of Variations by I.M. Gelfand; S.V. Fomin",
        "filename": "calculus_variations_gelfand_fomin.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Calculus%20Of%20Variations%20by%20I.M.%20Gelfand;%20S.V.%20Fomin.pdf",
            "https://github.com/math-books/math-books/raw/main/Calculus%20Of%20Variations%20by%20I.M.%20Gelfand;%20S.V.%20Fomin.pdf"
        ]
    },
    
    # Optimization and Control
    {
        "title": "Optimal control by Alekseev, Tikhomirov and Fomin",
        "filename": "optimal_control_alekseev.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Optimal%20control%20by%20Alekseev,%20Tikhomirov%20and%20Fomin.pdf",
            "https://github.com/math-books/math-books/raw/main/Optimal%20control%20by%20Alekseev,%20Tikhomirov%20and%20Fomin.pdf"
        ]
    },
    {
        "title": "Operations research by Elena Wentzel",
        "filename": "operations_research_wentzel.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Operations%20research%20by%20Elena%20Wentzel.pdf",
            "https://github.com/math-books/math-books/raw/main/Operations%20research%20by%20Elena%20Wentzel.pdf"
        ]
    },
    
    # Probability and Statistics
    {
        "title": "Probability - First Steps by E.S. Wentzel",
        "filename": "probability_first_steps_wentzel.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Probability%20-%20First%20Steps%20by%20E.S.%20Wentzel.pdf",
            "https://github.com/math-books/math-books/raw/main/Probability%20-%20First%20Steps%20by%20E.S.%20Wentzel.pdf"
        ]
    },
    {
        "title": "Applied Problems in Probability Theory by Wentzel and Ovcharov",
        "filename": "applied_problems_probability_wentzel_ovcharov.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Applied%20Problems%20in%20Probability%20Theory%20by%20Wentzel%20and%20Ovcharov.pdf",
            "https://github.com/math-books/math-books/raw/main/Applied%20Problems%20in%20Probability%20Theory%20by%20Wentzel%20and%20Ovcharov.pdf"
        ]
    },
    
    # Machine Learning Mathematics
    {
        "title": "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
        "filename": "mathematics_machine_learning_deisenroth.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/MATHEMATICS%20FOR%20MACHINE%20LEARNING%20by%20Marc%20Peter%20Deisenroth.pdf",
            "https://github.com/math-books/math-books/raw/main/MATHEMATICS%20FOR%20MACHINE%20LEARNING%20by%20Marc%20Peter%20Deisenroth.pdf"
        ]
    },
    
    # Advanced Mathematical Analysis
    {
        "title": "Principles of Mathematical Analysis by Walter Rubin",
        "filename": "principles_mathematical_analysis_rubin.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Principles%20of%20Mathematical%20Analysis%20by%20Walter%20Rubin.pdf",
            "https://github.com/math-books/math-books/raw/main/Principles%20of%20Mathematical%20Analysis%20by%20Walter%20Rubin.pdf"
        ]
    },
    {
        "title": "Introduction to matrix analysis by Richard Bellman",
        "filename": "introduction_matrix_analysis_bellman.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Introduction%20to%20matrix%20analysis%20by%20Richard%20Bellman.pdf",
            "https://github.com/math-books/math-books/raw/main/Introduction%20to%20matrix%20analysis%20by%20Richard%20Bellman.pdf"
        ]
    },
    
    # Problem-Solving and Mathematical Thinking
    {
        "title": "Challenging mathematical problems with elementary solutions, Volume I, Combinatorial Analysis and Probability Theory Yaglom & Yaglom",
        "filename": "challenging_mathematical_problems_yaglom_vol1.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Challenging%20mathematical%20problems%20with%20elementary%20solutions%20Vol%201%20Yaglom%20%26%20Yaglom.pdf",
            "https://github.com/math-books/math-books/raw/main/Challenging%20mathematical%20problems%20with%20elementary%20solutions%20Vol%201%20Yaglom%20%26%20Yaglom.pdf"
        ]
    },
    {
        "title": "Collection of problems in probability theory by Meshalkin",
        "filename": "collection_problems_probability_meshalkin.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Collection%20of%20problems%20in%20probability%20theory%20by%20Meshalkin.pdf",
            "https://github.com/math-books/math-books/raw/main/Collection%20of%20problems%20in%20probability%20theory%20by%20Meshalkin.pdf"
        ]
    },
    
    # Advanced Topics
    {
        "title": "Kolmogorov Complexity and Algorithmic Randomness by Shen, Uspensky, Vereschagin",
        "filename": "kolmogorov_complexity_algorithmic_randomness.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/Kolmogorov%20Complexity%20and%20Algorithmic%20Randomness%20by%20Shen%20Uspensky%20Vereschagin.pdf",
            "https://github.com/math-books/math-books/raw/main/Kolmogorov%20Complexity%20and%20Algorithmic%20Randomness%20by%20Shen%20Uspensky%20Vereschagin.pdf"
        ]
    },
    {
        "title": "A Collection of Problems on a Course of Mathematical Analysis by Berman",
        "filename": "collection_problems_mathematical_analysis_berman.pdf",
        "urls": [
            "https://raw.githubusercontent.com/math-books/math-books/main/A%20Collection%20of%20Problems%20on%20a%20Course%20of%20Mathematical%20Analysis%20by%20Berman.pdf",
            "https://github.com/math-books/math-books/raw/main/A%20Collection%20of%20Problems%20on%20a%20Course%20of%20Mathematical%20Analysis%20by%20Berman.pdf"
        ]
    }
]

def download_file_with_fallback(urls, filename, download_dir="optimization_books"):
    """Download a file with multiple URL fallbacks"""
    try:
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        filepath = os.path.join(download_dir, filename)
        
        print(f"📖 Downloading: {filename}")
        
        # Try each URL until one works
        for i, url in enumerate(urls):
            try:
                print(f"   Attempt {i+1}: {url}")
                
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
                                print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                
                print(f"\n✅ Successfully downloaded: {filename}")
                print(f"   Size: {os.path.getsize(filepath)} bytes")
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Attempt {i+1} failed: {e}")
                if i < len(urls) - 1:
                    print("   Trying next URL...")
                    time.sleep(1)
                else:
                    print(f"   All URLs failed for {filename}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def categorize_books():
    """Categorize books by their optimization relevance"""
    
    print("📚 Optimization Book Categories")
    print("=" * 50)
    
    categories = {
        "Essential Optimization": [
            "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
            "Optimal control by Alekseev, Tikhomirov and Fomin",
            "Operations research by Elena Wentzel"
        ],
        "Matrix Analysis & Linear Algebra": [
            "Theory of matrices, Volume I by Gantmacher",
            "Theory of matrices, Volume II by Gantmacher",
            "Linear Algebra by V. V. Voyevodin",
            "Problems In Linear Algebra by I. V. Proskuryakov",
            "Introduction to matrix analysis by Richard Bellman"
        ],
        "Mathematical Analysis": [
            "A Course Of Mathematical Analysis by A. Ya. Khinchin",
            "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
            "Calculus Of Variations by I.M. Gelfand; S.V. Fomin",
            "Principles of Mathematical Analysis by Walter Rubin",
            "A Collection of Problems on a Course of Mathematical Analysis by Berman"
        ],
        "Probability & Statistics": [
            "Probability - First Steps by E.S. Wentzel",
            "Applied Problems in Probability Theory by Wentzel and Ovcharov",
            "Collection of problems in probability theory by Meshalkin"
        ],
        "Machine Learning Math": [
            "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong"
        ],
        "Problem Solving": [
            "Challenging mathematical problems with elementary solutions, Volume I, Combinatorial Analysis and Probability Theory Yaglom & Yaglom"
        ],
        "Advanced Topics": [
            "Kolmogorov Complexity and Algorithmic Randomness by Shen, Uspensky, Vereschagin"
        ]
    }
    
    for category, books in categories.items():
        print(f"\n🔧 {category}:")
        for book in books:
            print(f"   • {book}")
    
    return categories

def main():
    """Main download function"""
    
    print("🚀 Comprehensive Optimization Book Downloader")
    print("=" * 60)
    
    # Categorize books
    categories = categorize_books()
    
    print(f"\n📥 Starting downloads for {len(OPTIMIZATION_BOOKS)} books...")
    print("=" * 60)
    
    successful_downloads = 0
    total_downloads = len(OPTIMIZATION_BOOKS)
    
    for book in OPTIMIZATION_BOOKS:
        print(f"\n📖 {book['title']}")
        print("-" * 50)
        
        if download_file_with_fallback(book['urls'], book['filename']):
            successful_downloads += 1
        
        # Small delay between downloads
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"📊 Download Summary:")
    print(f"   Successful: {successful_downloads}/{total_downloads}")
    print(f"   Failed: {total_downloads - successful_downloads}/{total_downloads}")
    
    if successful_downloads > 0:
        print(f"\n📁 Books downloaded to: optimization_books/")
        print("\n🎯 These books will provide:")
        print("   • Advanced optimization algorithms")
        print("   • Matrix analysis for neural networks")
        print("   • Mathematical analysis techniques")
        print("   • Probability theory for ML")
        print("   • Problem-solving strategies")
        print("   • Machine learning mathematics")
        
        print("\n📚 Book Organization:")
        print("   • Essential optimization theory")
        print("   • Linear algebra and matrix operations")
        print("   • Mathematical analysis and calculus")
        print("   • Probability and statistics")
        print("   • Machine learning foundations")
        print("   • Advanced mathematical topics")
        
        print("\n🚀 Next Steps:")
        print("   1. Study these books for optimization insights")
        print("   2. Apply mathematical principles to FFN-v2 training")
        print("   3. Use matrix analysis for neural network optimization")
        print("   4. Implement advanced optimization algorithms")

if __name__ == "__main__":
    main() 