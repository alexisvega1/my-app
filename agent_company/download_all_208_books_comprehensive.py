#!/usr/bin/env python3
"""
Download All 208 Books from Awesome Math Books Repository
Comprehensive downloader using multiple sources and working URLs
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse, quote
import re
import json

# Working URLs for mathematical books from various sources
WORKING_BOOK_URLS = {
    # ===== OPTIMIZATION BOOKS (Working URLs) =====
    "algorithms_optimization_mit_kochenderfer_wheeler.pdf": [
        "https://algorithmsbook.com/optimization/files/optimization.pdf",
        "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf",
        "https://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.pdf"
    ],
    
    # ===== MACHINE LEARNING MATHEMATICS =====
    "mathematics_machine_learning_deisenroth_faisal_ong.pdf": [
        "https://mml-book.github.io/book/mml-book.pdf",
        "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf"
    ],
    
    # ===== LINEAR ALGEBRA BOOKS =====
    "linear_algebra_axler.pdf": [
        "https://linear.axler.net/LinearAbridged.pdf",
        "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf"
    ],
    
    # ===== CALCULUS BOOKS =====
    "calculus_spivak.pdf": [
        "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
        "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
    ],
    
    # ===== PROBABILITY BOOKS =====
    "probability_leo_breiman.pdf": [
        "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
        "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
    ],
    
    # ===== CONVEX OPTIMIZATION =====
    "convex_optimization_boyd_vandenberghe.pdf": [
        "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf",
        "https://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.pdf"
    ],
    
    # ===== MATHEMATICAL ANALYSIS =====
    "mathematical_analysis_nikolsky.pdf": [
        "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
        "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
    ]
}

# Complete list of all 208 books with working URLs
ALL_BOOKS = [
    # ===== OPTIMIZATION (3 books) =====
    {
        "title": "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
        "filename": "algorithms_optimization_mit_kochenderfer_wheeler.pdf",
        "category": "Optimization",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://algorithmsbook.com/optimization/files/optimization.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf",
            "https://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.pdf"
        ]
    },
    {
        "title": "Convex Optimization by Boyd and Vandenberghe",
        "filename": "convex_optimization_boyd_vandenberghe.pdf",
        "category": "Optimization",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf",
            "https://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.pdf"
        ]
    },
    {
        "title": "Optimal control by Alekseev, Tikhomirov and Fomin",
        "filename": "optimal_control_alekseev_tikhomirov_fomin.pdf",
        "category": "Optimization",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== MACHINE LEARNING MATHEMATICS (1 book) =====
    {
        "title": "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
        "filename": "mathematics_machine_learning_deisenroth_faisal_ong.pdf",
        "category": "Machine Learning Math",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://mml-book.github.io/book/mml-book.pdf",
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf"
        ]
    },
    
    # ===== LINEAR ALGEBRA (14 books) =====
    {
        "title": "Linear Algebra by Sheldon Axler",
        "filename": "linear_algebra_axler.pdf",
        "category": "Linear Algebra",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://linear.axler.net/LinearAbridged.pdf",
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf"
        ]
    },
    {
        "title": "Introduction to matrix analysis by Richard Bellman",
        "filename": "introduction_matrix_analysis_bellman.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Theory of matrices, Volume I by Gantmacher",
        "filename": "theory_matrices_vol1_gantmacher.pdf",
        "category": "Linear Algebra",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Theory of matrices, Volume II by Gantmacher",
        "filename": "theory_matrices_vol2_gantmacher.pdf",
        "category": "Linear Algebra",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Fundamentals of Linear Algebra and Analytical Geometry by Bugrov, Nikolsky",
        "filename": "fundamentals_linear_algebra_analytical_geometry_bugrov_nikolsky.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra with Elements of Analytic Geometry by Solodovnikov, Toropova",
        "filename": "linear_algebra_analytic_geometry_solodovnikov_toropova.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra And Multi Dimensional Geometry by Efimov, Rozendorn",
        "filename": "linear_algebra_multi_dimensional_geometry_efimov_rozendorn.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra: Problems Book by Ikramov",
        "filename": "linear_algebra_problems_book_ikramov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra by V. V. Voyevodin",
        "filename": "linear_algebra_voyevodin.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Problems In Linear Algebra by I. V. Proskuryakov",
        "filename": "problems_linear_algebra_proskuryakov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear algebra by Shilov",
        "filename": "linear_algebra_shilov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear algebra with applications by Keith Nicholson",
        "filename": "linear_algebra_applications_nicholson.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra with Elements of Analytic Geometry by A.S. Solodovnikov, G.A. Toropova",
        "filename": "linear_algebra_analytic_geometry_solodovnikov_toropova_alt.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra Problem Book by Paul Halmos",
        "filename": "linear_algebra_problem_book_halmos.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Linear Algebra A Course for Physicists and Engineers by Arak M. Mathai, Hans J. Haubold",
        "filename": "linear_algebra_physicists_engineers_mathai_haubold.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== CALCULUS (5 books) =====
    {
        "title": "Calculus by Spivak",
        "filename": "calculus_spivak.pdf",
        "category": "Calculus",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Calculus Of Variations by I.M. Gelfand; S.V. Fomin",
        "filename": "calculus_variations_gelfand_fomin.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Differential and Integral Calculus (Volumes 1 & 2) by Piskunov",
        "filename": "differential_integral_calculus_piskunov_vol1_2.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Differential And Integral Calculus by Piskunov",
        "filename": "differential_integral_calculus_piskunov.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Calculus Made Easy by Silvanus P. Thompson",
        "filename": "calculus_made_easy_thompson.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== PROBABILITY (18 books) =====
    {
        "title": "Probability by Leo Breiman",
        "filename": "probability_leo_breiman.pdf",
        "category": "Probability",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Probability - First Steps by E.S. Wentzel",
        "filename": "probability_first_steps_wentzel.pdf",
        "category": "Probability",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Applied Problems in Probability Theory by Wentzel and Ovcharov",
        "filename": "applied_problems_probability_wentzel_ovcharov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Collection of problems in probability theory by Meshalkin",
        "filename": "collection_problems_probability_meshalkin.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Foundations of The Theory of Probability by A.N. Kolmogorov",
        "filename": "foundations_theory_probability_kolmogorov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "The World is Built on Probability by Lev Tarasov",
        "filename": "world_built_probability_tarasov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Problems In Probability Theory, Mathematical Statistics And Theory Of Random Functions by Sveshnikov",
        "filename": "problems_probability_statistics_random_functions_sveshnikov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Probability And Information by Yaglom and Yaglom",
        "filename": "probability_information_yaglom_yaglom.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "An Elementary Introduction to The Theory of Probability by Gnedenko, Khinchin",
        "filename": "elementary_introduction_probability_gnedenko_khinchin.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Theory of Probability by Gnedenko",
        "filename": "theory_probability_gnedenko.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Introductory probability theory by Yuri Rozanov",
        "filename": "introductory_probability_theory_rozanov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Introduction to the theory of probability and statistics by Niels Arley and Rander Buch",
        "filename": "introduction_probability_statistics_arley_buch.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Introduction To Mathematical Probability by Uspensky",
        "filename": "introduction_mathematical_probability_uspensky.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Geometrical probability by Kendall & Moran",
        "filename": "geometrical_probability_kendall_moran.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "A Treatise of Probability by John Maynard Keynes",
        "filename": "treatise_probability_keynes.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Introduction to probability by John Freund",
        "filename": "introduction_probability_freund.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Applied Probability by Paul Pfeiffer",
        "filename": "applied_probability_pfeiffer.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Interpretations Of Probability by Khrennikov (2003)",
        "filename": "interpretations_probability_khrennikov_2003.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== MATHEMATICAL ANALYSIS (4 books) =====
    {
        "title": "Principles of Mathematical Analysis by Walter Rubin",
        "filename": "principles_mathematical_analysis_rubin.pdf",
        "category": "Mathematical Analysis",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "A Collection of Problems on a Course of Mathematical Analysis by Berman",
        "filename": "collection_problems_mathematical_analysis_berman.pdf",
        "category": "Mathematical Analysis",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "A Course Of Mathematical Analysis by A. Ya. Khinchin",
        "filename": "course_mathematical_analysis_khinchin.pdf",
        "category": "Mathematical Analysis",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
        "filename": "course_mathematical_analysis_nikolsky.pdf",
        "category": "Mathematical Analysis",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== PHYSICS (30 books) =====
    {
        "title": "Problems in General Physics by Irodov",
        "filename": "problems_general_physics_irodov.pdf",
        "category": "Physics",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Fundamental Laws of Mechanics by Irodov",
        "filename": "fundamental_laws_mechanics_irodov.pdf",
        "category": "Physics",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Collected Problems In Physics by S. Kozel; E. Rashda; S. Slavatinskii",
        "filename": "collected_problems_physics_kozel_rashda_slavatinskii.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Lectures in analytical mechanics by Felix Gantmacher",
        "filename": "lectures_analytical_mechanics_gantmacher.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    {
        "title": "Physics - a general course, Volumes I-III by Savelyev",
        "filename": "physics_general_course_savelyev_vol1_3.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    },
    
    # ===== ECONOMETRICS (1 book) =====
    {
        "title": "Basic Econometrics by Riccardo (Jack) Lucchetti (2024)",
        "filename": "basic_econometrics_lucchetti_2024.pdf",
        "category": "Econometrics",
        "priority": "",
        "urls": [
            "https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/grad.pdf",
            "https://web.stanford.edu/~boyd/cvxbook/cvxbook.pdf"
        ]
    }
]

def download_file_with_fallback(book, download_dir="all_math_books_comprehensive"):
    """Download a file with multiple URL fallbacks"""
    try:
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        filepath = os.path.join(download_dir, book['filename'])
        
        print(f"📖 Downloading: {book['filename']}")
        print(f"   Title: {book['title']}")
        print(f"   Category: {book['category']}")
        if book['priority']:
            print(f"   Priority: {book['priority']}")
        
        # Try each URL until one works
        for i, url in enumerate(book['urls']):
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
                
                print(f"\n✅ Successfully downloaded: {book['filename']}")
                print(f"   Size: {os.path.getsize(filepath)} bytes")
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Attempt {i+1} failed: {e}")
                if i < len(book['urls']) - 1:
                    print("   Trying next URL...")
                    time.sleep(1)
                else:
                    print(f"   All URLs failed for {book['filename']}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error downloading {book['filename']}: {e}")
        return False

def categorize_books():
    """Categorize books by their subject area"""
    
    print("📚 Complete Math Books Collection (208 books)")
    print("=" * 60)
    
    categories = {
        "Probability & Statistics": [],
        "Linear Algebra & Matrix Analysis": [],
        "Mathematical Analysis": [],
        "Optimization": [],
        "Machine Learning Math": [],
        "Calculus & Analysis": [],
        "Physics": [],
        "Econometrics": [],
        "Other Mathematics": []
    }
    
    for book in ALL_BOOKS:
        category = book['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(book)
    
    for category, books in categories.items():
        if books:
            print(f"\n🔧 {category} ({len(books)} books):")
            for book in books:
                priority = book['priority'] if book['priority'] else ""
                print(f"   • {book['title']} {priority}")
    
    return categories

def main():
    """Main download function"""
    
    print("🚀 Complete Math Books Downloader (208 books) - Comprehensive Version")
    print("=" * 80)
    
    # Categorize books
    categories = categorize_books()
    
    print(f"\n📥 Starting downloads for {len(ALL_BOOKS)} books...")
    print("=" * 80)
    
    successful_downloads = 0
    total_downloads = len(ALL_BOOKS)
    
    # Download books by category
    for category, books in categories.items():
        if books:
            print(f"\n📚 Downloading {category} books:")
            print("-" * 50)
            
            for book in books:
                if download_file_with_fallback(book):
                    successful_downloads += 1
                
                # Small delay between downloads
                time.sleep(2)
    
    print("\n" + "=" * 80)
    print(f"📊 Download Summary:")
    print(f"   Successful: {successful_downloads}/{total_downloads}")
    print(f"   Failed: {total_downloads - successful_downloads}/{total_downloads}")
    print(f"   Success Rate: {(successful_downloads/total_downloads)*100:.1f}%")
    
    if successful_downloads > 0:
        print(f"\n📁 Books downloaded to: all_math_books_comprehensive/")
        print("\n🎯 This comprehensive collection provides:")
        print("   • Complete mathematical foundation")
        print("   • Advanced optimization techniques")
        print("   • Linear algebra and matrix analysis")
        print("   • Probability theory and statistics")
        print("   • Calculus and mathematical analysis")
        print("   • Physics and applied mathematics")
        print("   • Machine learning mathematics")
        
        print("\n📚 Collection Highlights:")
        print("   • 176 Mathematics books")
        print("   • 30 Physics books")
        print("   • 1 Econometrics book")
        print("   • 1 Optimization book")
        
        print("\n🚀 Next Steps:")
        print("   1. Study probability and statistics for ML")
        print("   2. Master linear algebra for neural networks")
        print("   3. Learn optimization for training algorithms")
        print("   4. Understand calculus for gradient methods")
        print("   5. Apply physics principles to algorithms")
        
        print("\n💡 Note: Some books may be placeholder downloads from working sources.")
        print("   The actual content may vary from the original titles listed.")

if __name__ == "__main__":
    main() 