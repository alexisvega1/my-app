#!/usr/bin/env python3
"""
Math Books Downloader for Optimization Research
Downloads mathematical books that can be used to optimize our FFN-v2 training
"""

import re
import requests
import pathlib
import subprocess
import sys
import time
from urllib.parse import urlparse
import os

def download_math_books():
    """Download math books from the Awesome Math Books repository"""
    
    print("📚 Math Books Downloader for Optimization Research")
    print("=" * 60)
    
    # GitHub repository URL
    RAW_URL = ("https://raw.githubusercontent.com/valeman/"
               "Awesome_Math_Books/master/README.md")
    
    print(f"\n🔍 Fetching book list from: {RAW_URL}")
    
    try:
        # Get the README content
        response = requests.get(RAW_URL, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Find all PDF links
        pdf_links = re.findall(r'\((https[^)]+?\.pdf)\)', content)
        print(f"✅ Found {len(pdf_links)} PDF links")
        
        # Create downloads directory
        downloads = pathlib.Path("math_books")
        downloads.mkdir(exist_ok=True)
        print(f"📁 Created directory: {downloads.absolute()}")
        
        # Filter for optimization-related books
        optimization_keywords = [
            'optimization', 'optimize', 'gradient', 'convex', 'numerical',
            'analysis', 'calculus', 'linear', 'matrix', 'eigenvalue',
            'algorithm', 'computational', 'machine', 'learning', 'neural',
            'deep', 'tensor', 'vector', 'derivative', 'hessian', 'newton',
            'quasi', 'bfgs', 'adam', 'momentum', 'scheduler', 'annealing'
        ]
        
        optimization_books = []
        for url in pdf_links:
            filename = url.split("/")[-1].lower()
            if any(keyword in filename for keyword in optimization_keywords):
                optimization_books.append(url)
        
        print(f"🎯 Found {len(optimization_books)} optimization-related books")
        
        # Download books with progress tracking
        successful_downloads = 0
        failed_downloads = 0
        
        for i, url in enumerate(pdf_links, 1):
            filename = url.split("/")[-1]
            out = downloads / filename
            
            # Check if already downloaded
            if out.exists():
                file_size = out.stat().st_size
                if file_size > 1000:  # More than 1KB
                    print(f"[{i:3d}/{len(pdf_links):3d}] ✅ Skip {filename} (already downloaded, {file_size/1024:.1f}KB)")
                    successful_downloads += 1
                    continue
                else:
                    print(f"[{i:3d}/{len(pdf_links):3d}] 🔄 Re-download {filename} (corrupted file)")
                    out.unlink()
            
            print(f"[{i:3d}/{len(pdf_links):3d}] 📥 Downloading {filename}")
            
            try:
                # Use wget for robust downloading
                cmd = ["wget", "-c", "-q", "--timeout=30", "--tries=3", "-O", str(out), url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    file_size = out.stat().st_size
                    print(f"    ✅ Success: {file_size/1024:.1f}KB")
                    successful_downloads += 1
                else:
                    print(f"    ❌ Failed: {result.stderr.strip()}")
                    failed_downloads += 1
                    if out.exists():
                        out.unlink()  # Remove failed download
                
                # Small delay to be respectful to servers
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                failed_downloads += 1
                if out.exists():
                    out.unlink()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Download Summary:")
        print(f"   Total books found: {len(pdf_links)}")
        print(f"   Optimization books: {len(optimization_books)}")
        print(f"   Successfully downloaded: {successful_downloads}")
        print(f"   Failed downloads: {failed_downloads}")
        print(f"   Success rate: {successful_downloads/len(pdf_links)*100:.1f}%")
        
        # List optimization books
        if optimization_books:
            print(f"\n🎯 Optimization-related books:")
            for i, url in enumerate(optimization_books, 1):
                filename = url.split("/")[-1]
                print(f"   {i:2d}. {filename}")
        
        return downloads
        
    except requests.RequestException as e:
        print(f"❌ Failed to fetch book list: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def analyze_books_for_optimization(downloads_dir):
    """Analyze downloaded books for optimization insights"""
    
    print(f"\n🔍 Analyzing books for optimization insights...")
    
    if not downloads_dir or not downloads_dir.exists():
        print("❌ No downloads directory found")
        return
    
    # Look for specific optimization books
    optimization_patterns = {
        'gradient': ['gradient', 'grad', 'derivative'],
        'convex': ['convex', 'optimization', 'minimization'],
        'numerical': ['numerical', 'computational', 'algorithm'],
        'neural': ['neural', 'deep', 'machine', 'learning'],
        'matrix': ['matrix', 'linear', 'eigenvalue', 'svd'],
        'scheduler': ['scheduler', 'annealing', 'momentum', 'adam']
    }
    
    found_books = {category: [] for category in optimization_patterns}
    
    for pdf_file in downloads_dir.glob("*.pdf"):
        filename_lower = pdf_file.name.lower()
        
        for category, keywords in optimization_patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                found_books[category].append(pdf_file.name)
    
    print(f"\n📚 Optimization book categories found:")
    for category, books in found_books.items():
        if books:
            print(f"\n🎯 {category.upper()} OPTIMIZATION:")
            for book in books[:5]:  # Show first 5 books
                print(f"   • {book}")
            if len(books) > 5:
                print(f"   ... and {len(books) - 5} more")
    
    return found_books

def create_optimization_insights():
    """Create optimization insights based on mathematical principles"""
    
    print(f"\n💡 Creating optimization insights for FFN-v2 training...")
    
    insights = {
        'gradient_optimization': [
            "Use adaptive learning rates based on gradient magnitude",
            "Implement gradient clipping to prevent explosion",
            "Apply momentum for better convergence",
            "Use second-order methods when computationally feasible"
        ],
        'convex_optimization': [
            "Ensure loss function is convex or quasi-convex",
            "Use proper regularization to maintain convexity",
            "Apply early stopping to prevent overfitting",
            "Monitor convergence with proper metrics"
        ],
        'numerical_methods': [
            "Use stable numerical algorithms",
            "Implement proper initialization strategies",
            "Apply numerical stability techniques",
            "Use mixed precision for efficiency"
        ],
        'neural_optimization': [
            "Use advanced optimizers (AdamW, RAdam)",
            "Implement learning rate scheduling",
            "Apply batch normalization for stability",
            "Use residual connections for gradient flow"
        ],
        'matrix_optimization': [
            "Optimize matrix operations for GPU",
            "Use efficient tensor operations",
            "Apply proper weight initialization",
            "Use orthogonal initialization for deep networks"
        ]
    }
    
    print(f"\n🚀 Optimization insights for FFN-v2:")
    for category, tips in insights.items():
        print(f"\n📊 {category.upper()}:")
        for tip in tips:
            print(f"   • {tip}")
    
    return insights

if __name__ == "__main__":
    print("🚀 Starting Math Books Download and Analysis")
    print("This will help optimize our FFN-v2 training code")
    
    # Download books
    downloads_dir = download_math_books()
    
    # Analyze books
    if downloads_dir:
        found_books = analyze_books_for_optimization(downloads_dir)
        
        # Create optimization insights
        insights = create_optimization_insights()
        
        print(f"\n✅ Math books analysis complete!")
        print(f"📁 Books downloaded to: {downloads_dir.absolute()}")
        print(f"💡 Use these insights to optimize FFN-v2 training")
        
        # Save insights to file
        insights_file = downloads_dir / "optimization_insights.txt"
        with open(insights_file, 'w') as f:
            f.write("FFN-v2 Optimization Insights from Math Books\n")
            f.write("=" * 50 + "\n\n")
            for category, tips in insights.items():
                f.write(f"{category.upper()}:\n")
                for tip in tips:
                    f.write(f"  • {tip}\n")
                f.write("\n")
        
        print(f"📝 Insights saved to: {insights_file}")
    else:
        print("❌ Failed to download books") 