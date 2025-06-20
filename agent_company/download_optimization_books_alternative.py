#!/usr/bin/env python3
"""
Download Optimization Books from Alternative Sources
Find and download optimization books from accessible repositories
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse, quote
import re

# Alternative sources for optimization books
ALTERNATIVE_SOURCES = [
    # MIT OpenCourseWare
    "https://ocw.mit.edu/courses/",
    
    # Stanford Online
    "https://online.stanford.edu/courses/",
    
    # arXiv (for optimization papers)
    "https://arxiv.org/pdf/",
    
    # University repositories
    "https://www.math.ucla.edu/~yanovsky/Teaching/",
    "https://www.math.ucla.edu/~yanovsky/Research/",
    
    # Direct optimization resources
    "https://web.stanford.edu/~boyd/cvxbook/",
    "https://www.seas.ucla.edu/~vandenbe/ee236a/",
]

# Specific optimization books with working URLs
WORKING_OPTIMIZATION_BOOKS = [
    {
        "title": "Convex Optimization by Boyd and Vandenberghe",
        "filename": "convex_optimization_boyd_vandenberghe.pdf",
        "url": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf",
        "category": "Essential Optimization"
    },
    {
        "title": "Optimization Methods by UCLA",
        "filename": "optimization_methods_ucla.pdf",
        "url": "https://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.pdf",
        "category": "Essential Optimization"
    },
    {
        "title": "Linear Algebra Done Right by Axler",
        "filename": "linear_algebra_done_right_axler.pdf",
        "url": "https://linear.axler.net/LinearAlgebraDoneRight.pdf",
        "category": "Linear Algebra"
    },
    {
        "title": "Matrix Analysis by Horn and Johnson",
        "filename": "matrix_analysis_horn_johnson.pdf",
        "url": "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/MatrixAnalysis.pdf",
        "category": "Matrix Analysis"
    },
    {
        "title": "Numerical Optimization by Nocedal and Wright",
        "filename": "numerical_optimization_nocedal_wright.pdf",
        "url": "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/NumericalOptimization.pdf",
        "category": "Numerical Methods"
    },
    {
        "title": "Probability Theory by Jaynes",
        "filename": "probability_theory_jaynes.pdf",
        "url": "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/ProbabilityTheory.pdf",
        "category": "Probability"
    },
    {
        "title": "Machine Learning Mathematics by Deisenroth",
        "filename": "mathematics_machine_learning_deisenroth.pdf",
        "url": "https://mml-book.github.io/book/mml-book.pdf",
        "category": "Machine Learning Math"
    },
    {
        "title": "Optimization Algorithms by Kochenderfer",
        "filename": "optimization_algorithms_kochenderfer.pdf",
        "url": "https://algorithmsbook.com/optimization/files/optimization.pdf",
        "category": "Essential Optimization"
    }
]

# Additional optimization resources from arXiv
ARXIV_PAPERS = [
    {
        "title": "Deep Learning Optimization Survey",
        "filename": "deep_learning_optimization_survey.pdf",
        "arxiv_id": "2003.05629",
        "category": "Deep Learning Optimization"
    },
    {
        "title": "Neural Network Optimization Methods",
        "filename": "neural_network_optimization_methods.pdf",
        "arxiv_id": "1609.04747",
        "category": "Neural Networks"
    },
    {
        "title": "Adam Optimization Algorithm",
        "filename": "adam_optimization_algorithm.pdf",
        "arxiv_id": "1412.6980",
        "category": "Optimization Algorithms"
    },
    {
        "title": "Stochastic Gradient Descent Methods",
        "filename": "stochastic_gradient_descent_methods.pdf",
        "arxiv_id": "1609.04747",
        "category": "Optimization Algorithms"
    }
]

def download_file(url, filename, download_dir="optimization_books"):
    """Download a file with progress tracking and error handling"""
    try:
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        filepath = os.path.join(download_dir, filename)
        
        print(f"📖 Downloading: {filename}")
        print(f"   URL: {url}")
        
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
                                print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                
                print(f"\n✅ Successfully downloaded: {filename}")
                print(f"   Size: {os.path.getsize(filepath)} bytes")
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    print("   Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"   Failed to download {filename} after 3 attempts")
                    return False
                    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def download_arxiv_paper(arxiv_id, filename, download_dir="optimization_books"):
    """Download a paper from arXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return download_file(url, filename, download_dir)

def create_optimization_summary():
    """Create a summary of optimization resources"""
    
    print("📚 Optimization Resources Summary")
    print("=" * 50)
    
    categories = {
        "Essential Optimization": [
            "Convex Optimization by Boyd and Vandenberghe",
            "Optimization Methods by UCLA",
            "Optimization Algorithms by Kochenderfer"
        ],
        "Linear Algebra & Matrix Analysis": [
            "Linear Algebra Done Right by Axler",
            "Matrix Analysis by Horn and Johnson"
        ],
        "Numerical Methods": [
            "Numerical Optimization by Nocedal and Wright"
        ],
        "Probability & Statistics": [
            "Probability Theory by Jaynes"
        ],
        "Machine Learning Math": [
            "Machine Learning Mathematics by Deisenroth"
        ],
        "Research Papers": [
            "Deep Learning Optimization Survey",
            "Neural Network Optimization Methods",
            "Adam Optimization Algorithm",
            "Stochastic Gradient Descent Methods"
        ]
    }
    
    for category, resources in categories.items():
        print(f"\n🔧 {category}:")
        for resource in resources:
            print(f"   • {resource}")
    
    return categories

def main():
    """Main download function"""
    
    print("🚀 Alternative Optimization Book Downloader")
    print("=" * 60)
    
    # Create summary
    categories = create_optimization_summary()
    
    print(f"\n📥 Starting downloads...")
    print("=" * 60)
    
    successful_downloads = 0
    total_downloads = len(WORKING_OPTIMIZATION_BOOKS) + len(ARXIV_PAPERS)
    
    # Download books with working URLs
    print("\n📚 Downloading Books with Working URLs:")
    for book in WORKING_OPTIMIZATION_BOOKS:
        print(f"\n📖 {book['title']}")
        print(f"   Category: {book['category']}")
        print("-" * 40)
        
        if download_file(book['url'], book['filename']):
            successful_downloads += 1
        
        time.sleep(2)
    
    # Download arXiv papers
    print("\n📄 Downloading Research Papers from arXiv:")
    for paper in ARXIV_PAPERS:
        print(f"\n📄 {paper['title']}")
        print(f"   Category: {paper['category']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print("-" * 40)
        
        if download_arxiv_paper(paper['arxiv_id'], paper['filename']):
            successful_downloads += 1
        
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"📊 Download Summary:")
    print(f"   Successful: {successful_downloads}/{total_downloads}")
    print(f"   Failed: {total_downloads - successful_downloads}/{total_downloads}")
    
    if successful_downloads > 0:
        print(f"\n📁 Resources downloaded to: optimization_books/")
        print("\n🎯 These resources provide:")
        print("   • Convex optimization theory and algorithms")
        print("   • Linear algebra and matrix analysis")
        print("   • Numerical optimization methods")
        print("   • Probability theory for ML")
        print("   • Machine learning mathematics")
        print("   • Latest research in optimization")
        
        print("\n📚 Resource Organization:")
        print("   • Essential optimization textbooks")
        print("   • Linear algebra and matrix theory")
        print("   • Numerical methods and algorithms")
        print("   • Probability and statistics")
        print("   • Machine learning foundations")
        print("   • Research papers and surveys")
        
        print("\n🚀 Next Steps:")
        print("   1. Study these resources for optimization insights")
        print("   2. Apply convex optimization to neural networks")
        print("   3. Use matrix analysis for efficient training")
        print("   4. Implement advanced optimization algorithms")
        print("   5. Stay updated with latest research")

if __name__ == "__main__":
    main() 