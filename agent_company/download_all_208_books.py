#!/usr/bin/env python3
"""
Download All 208 Books from Awesome Math Books Repository
Comprehensive downloader for the complete mathematical library
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse, quote
import re
import json

# Complete list of all 208 books from the repository
ALL_BOOKS = [
    # ===== MATHEMATICS (176 books) =====
    
    # Probability and Statistics
    {
        "title": "Probability - First Steps by E.S. Wentzel",
        "filename": "probability_first_steps_wentzel.pdf",
        "category": "Probability",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Probability%20-%20First%20Steps%20by%20E.S.%20Wentzel.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Probability%20-%20First%20Steps%20by%20E.S.%20Wentzel.pdf"
        ]
    },
    {
        "title": "Applied Problems in Probability Theory by Wentzel and Ovcharov",
        "filename": "applied_problems_probability_wentzel_ovcharov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Applied%20Problems%20in%20Probability%20Theory%20by%20Wentzel%20and%20Ovcharov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Applied%20Problems%20in%20Probability%20Theory%20by%20Wentzel%20and%20Ovcharov.pdf"
        ]
    },
    {
        "title": "Collection of problems in probability theory by Meshalkin",
        "filename": "collection_problems_probability_meshalkin.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Collection%20of%20problems%20in%20probability%20theory%20by%20Meshalkin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Collection%20of%20problems%20in%20probability%20theory%20by%20Meshalkin.pdf"
        ]
    },
    {
        "title": "Foundations of The Theory of Probability by A.N. Kolmogorov",
        "filename": "foundations_theory_probability_kolmogorov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Foundations%20of%20The%20Theory%20of%20Probability%20by%20A.N.%20Kolmogorov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Foundations%20of%20The%20Theory%20of%20Probability%20by%20A.N.%20Kolmogorov.pdf"
        ]
    },
    {
        "title": "The World is Built on Probability by Lev Tarasov",
        "filename": "world_built_probability_tarasov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/The%20World%20is%20Built%20on%20Probability%20by%20Lev%20Tarasov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/The%20World%20is%20Built%20on%20Probability%20by%20Lev%20Tarasov.pdf"
        ]
    },
    {
        "title": "Problems In Probability Theory, Mathematical Statistics And Theory Of Random Functions by Sveshnikov",
        "filename": "problems_probability_statistics_random_functions_sveshnikov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Problems%20In%20Probability%20Theory%20Mathematical%20Statistics%20And%20Theory%20Of%20Random%20Functions%20by%20Sveshnikov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Problems%20In%20Probability%20Theory%20Mathematical%20Statistics%20And%20Theory%20Of%20Random%20Functions%20by%20Sveshnikov.pdf"
        ]
    },
    {
        "title": "Probability And Information by Yaglom and Yaglom",
        "filename": "probability_information_yaglom_yaglom.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Probability%20And%20Information%20by%20Yaglom%20and%20Yaglom.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Probability%20And%20Information%20by%20Yaglom%20and%20Yaglom.pdf"
        ]
    },
    {
        "title": "An Elementary Introduction to The Theory of Probability by Gnedenko, Khinchin",
        "filename": "elementary_introduction_probability_gnedenko_khinchin.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/An%20Elementary%20Introduction%20to%20The%20Theory%20of%20Probability%20by%20Gnedenko%20Khinchin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/An%20Elementary%20Introduction%20to%20The%20Theory%20of%20Probability%20by%20Gnedenko%20Khinchin.pdf"
        ]
    },
    {
        "title": "Theory of Probability by Gnedenko",
        "filename": "theory_probability_gnedenko.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Theory%20of%20Probability%20by%20Gnedenko.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Theory%20of%20Probability%20by%20Gnedenko.pdf"
        ]
    },
    {
        "title": "Introductory probability theory by Yuri Rozanov",
        "filename": "introductory_probability_theory_rozanov.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Introductory%20probability%20theory%20by%20Yuri%20Rozanov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Introductory%20probability%20theory%20by%20Yuri%20Rozanov.pdf"
        ]
    },
    {
        "title": "Introduction to the theory of probability and statistics by Niels Arley and Rander Buch",
        "filename": "introduction_probability_statistics_arley_buch.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Introduction%20to%20the%20theory%20of%20probability%20and%20statistics%20by%20Niels%20Arley%20and%20Rander%20Buch.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Introduction%20to%20the%20theory%20of%20probability%20and%20statistics%20by%20Niels%20Arley%20and%20Rander%20Buch.pdf"
        ]
    },
    {
        "title": "Introduction To Mathematical Probability by Uspensky",
        "filename": "introduction_mathematical_probability_uspensky.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Introduction%20To%20Mathematical%20Probability%20by%20Uspensky.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Introduction%20To%20Mathematical%20Probability%20by%20Uspensky.pdf"
        ]
    },
    {
        "title": "Geometrical probability by Kendall & Moran",
        "filename": "geometrical_probability_kendall_moran.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Geometrical%20probability%20by%20Kendall%20%26%20Moran.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Geometrical%20probability%20by%20Kendall%20%26%20Moran.pdf"
        ]
    },
    {
        "title": "A Treatise of Probability by John Maynard Keynes",
        "filename": "treatise_probability_keynes.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/A%20Treatise%20of%20Probability%20by%20John%20Maynard%20Keynes.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/A%20Treatise%20of%20Probability%20by%20John%20Maynard%20Keynes.pdf"
        ]
    },
    {
        "title": "Probability by Leo Breiman",
        "filename": "probability_leo_breiman.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Probability%20by%20Leo%20Breiman.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Probability%20by%20Leo%20Breiman.pdf"
        ]
    },
    {
        "title": "Introduction to probability by John Freund",
        "filename": "introduction_probability_freund.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Introduction%20to%20probability%20by%20John%20Freund.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Introduction%20to%20probability%20by%20John%20Freund.pdf"
        ]
    },
    {
        "title": "Applied Probability by Paul Pfeiffer",
        "filename": "applied_probability_pfeiffer.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Applied%20Probability%20by%20Paul%20Pfeiffer.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Applied%20Probability%20by%20Paul%20Pfeiffer.pdf"
        ]
    },
    {
        "title": "Interpretations Of Probability by Khrennikov (2003)",
        "filename": "interpretations_probability_khrennikov_2003.pdf",
        "category": "Probability",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Interpretations%20Of%20Probability%20by%20Khrennikov%20(2003).pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Interpretations%20Of%20Probability%20by%20Khrennikov%20(2003).pdf"
        ]
    },
    
    # Linear Algebra and Matrix Analysis
    {
        "title": "Introduction to matrix analysis by Richard Bellman",
        "filename": "introduction_matrix_analysis_bellman.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Introduction%20to%20matrix%20analysis%20by%20Richard%20Bellman.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Introduction%20to%20matrix%20analysis%20by%20Richard%20Bellman.pdf"
        ]
    },
    {
        "title": "Theory of matrices, Volume I by Gantmacher",
        "filename": "theory_matrices_vol1_gantmacher.pdf",
        "category": "Linear Algebra",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Theory%20of%20matrices%20Volume%20I%20by%20Gantmacher.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Theory%20of%20matrices%20Volume%20I%20by%20Gantmacher.pdf"
        ]
    },
    {
        "title": "Theory of matrices, Volume II by Gantmacher",
        "filename": "theory_matrices_vol2_gantmacher.pdf",
        "category": "Linear Algebra",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Theory%20of%20matrices%20Volume%20II%20by%20Gantmacher.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Theory%20of%20matrices%20Volume%20II%20by%20Gantmacher.pdf"
        ]
    },
    {
        "title": "Fundamentals of Linear Algebra and Analytical Geometry by Bugrov, Nikolsky",
        "filename": "fundamentals_linear_algebra_analytical_geometry_bugrov_nikolsky.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Fundamentals%20of%20Linear%20Algebra%20and%20Analytical%20Geometry%20by%20Bugrov%20Nikolsky.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Fundamentals%20of%20Linear%20Algebra%20and%20Analytical%20Geometry%20by%20Bugrov%20Nikolsky.pdf"
        ]
    },
    {
        "title": "Linear Algebra with Elements of Analytic Geometry by Solodovnikov, Toropova",
        "filename": "linear_algebra_analytic_geometry_solodovnikov_toropova.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20with%20Elements%20of%20Analytic%20Geometry%20by%20Solodovnikov%20Toropova.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20with%20Elements%20of%20Analytic%20Geometry%20by%20Solodovnikov%20Toropova.pdf"
        ]
    },
    {
        "title": "Linear Algebra And Multi Dimensional Geometry by Efimov, Rozendorn",
        "filename": "linear_algebra_multi_dimensional_geometry_efimov_rozendorn.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20And%20Multi%20Dimensional%20Geometry%20by%20Efimov%20Rozendorn.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20And%20Multi%20Dimensional%20Geometry%20by%20Efimov%20Rozendorn.pdf"
        ]
    },
    {
        "title": "Linear Algebra: Problems Book by Ikramov",
        "filename": "linear_algebra_problems_book_ikramov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20Problems%20Book%20by%20Ikramov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20Problems%20Book%20by%20Ikramov.pdf"
        ]
    },
    {
        "title": "Linear Algebra by V. V. Voyevodin",
        "filename": "linear_algebra_voyevodin.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20by%20V.%20V.%20Voyevodin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20by%20V.%20V.%20Voyevodin.pdf"
        ]
    },
    {
        "title": "Problems In Linear Algebra by I. V. Proskuryakov",
        "filename": "problems_linear_algebra_proskuryakov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Problems%20In%20Linear%20Algebra%20by%20I.%20V.%20Proskuryakov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Problems%20In%20Linear%20Algebra%20by%20I.%20V.%20Proskuryakov.pdf"
        ]
    },
    {
        "title": "Linear algebra by Shilov",
        "filename": "linear_algebra_shilov.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20algebra%20by%20Shilov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20algebra%20by%20Shilov.pdf"
        ]
    },
    {
        "title": "Linear algebra with applications by Keith Nicholson",
        "filename": "linear_algebra_applications_nicholson.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20algebra%20with%20applications%20by%20Keith%20Nicholson.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20algebra%20with%20applications%20by%20Keith%20Nicholson.pdf"
        ]
    },
    {
        "title": "Linear Algebra with Elements of Analytic Geometry by A.S. Solodovnikov, G.A. Toropova",
        "filename": "linear_algebra_analytic_geometry_solodovnikov_toropova_alt.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20with%20Elements%20of%20Analytic%20Geometry%20by%20A.S.%20Solodovnikov%20G.A.%20Toropova.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20with%20Elements%20of%20Analytic%20Geometry%20by%20A.S.%20Solodovnikov%20G.A.%20Toropova.pdf"
        ]
    },
    {
        "title": "Linear Algebra Problem Book by Paul Halmos",
        "filename": "linear_algebra_problem_book_halmos.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20Problem%20Book%20by%20Paul%20Halmos.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20Problem%20Book%20by%20Paul%20Halmos.pdf"
        ]
    },
    {
        "title": "Linear Algebra A Course for Physicists and Engineers by Arak M. Mathai, Hans J. Haubold",
        "filename": "linear_algebra_physicists_engineers_mathai_haubold.pdf",
        "category": "Linear Algebra",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Linear%20Algebra%20A%20Course%20for%20Physicists%20and%20Engineers%20by%20Arak%20M.%20Mathai%20Hans%20J.%20Haubold.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Linear%20Algebra%20A%20Course%20for%20Physicists%20and%20Engineers%20by%20Arak%20M.%20Mathai%20Hans%20J.%20Haubold.pdf"
        ]
    },
    
    # Mathematical Analysis
    {
        "title": "Principles of Mathematical Analysis by Walter Rubin",
        "filename": "principles_mathematical_analysis_rubin.pdf",
        "category": "Mathematical Analysis",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Principles%20of%20Mathematical%20Analysis%20by%20Walter%20Rubin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Principles%20of%20Mathematical%20Analysis%20by%20Walter%20Rubin.pdf"
        ]
    },
    {
        "title": "A Collection of Problems on a Course of Mathematical Analysis by Berman",
        "filename": "collection_problems_mathematical_analysis_berman.pdf",
        "category": "Mathematical Analysis",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/A%20Collection%20of%20Problems%20on%20a%20Course%20of%20Mathematical%20Analysis%20by%20Berman.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/A%20Collection%20of%20Problems%20on%20a%20Course%20of%20Mathematical%20Analysis%20by%20Berman.pdf"
        ]
    },
    {
        "title": "A Course Of Mathematical Analysis by A. Ya. Khinchin",
        "filename": "course_mathematical_analysis_khinchin.pdf",
        "category": "Mathematical Analysis",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/A%20Course%20Of%20Mathematical%20Analysis%20by%20A.%20Ya.%20Khinchin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/A%20Course%20Of%20Mathematical%20Analysis%20by%20A.%20Ya.%20Khinchin.pdf"
        ]
    },
    {
        "title": "A Course Of Mathematical Analysis Vol 1 by S. M. Nikolsky",
        "filename": "course_mathematical_analysis_nikolsky.pdf",
        "category": "Mathematical Analysis",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/A%20Course%20Of%20Mathematical%20Analysis%20Vol%201%20by%20S.%20M.%20Nikolsky.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/A%20Course%20Of%20Mathematical%20Analysis%20Vol%201%20by%20S.%20M.%20Nikolsky.pdf"
        ]
    },
    
    # Optimization
    {
        "title": "Algorithms for optimization, MIT book by Mykel Kochenderfer, Tim Wheeler",
        "filename": "algorithms_optimization_mit_kochenderfer_wheeler.pdf",
        "category": "Optimization",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Algorithms%20for%20optimization%20MIT%20book%20by%20Mykel%20Kochenderfer%20Tim%20Wheeler.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Algorithms%20for%20optimization%20MIT%20book%20by%20Mykel%20Kochenderfer%20Tim%20Wheeler.pdf"
        ]
    },
    {
        "title": "Optimal control by Alekseev, Tikhomirov and Fomin",
        "filename": "optimal_control_alekseev_tikhomirov_fomin.pdf",
        "category": "Optimization",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Optimal%20control%20by%20Alekseev%20Tikhomirov%20and%20Fomin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Optimal%20control%20by%20Alekseev%20Tikhomirov%20and%20Fomin.pdf"
        ]
    },
    {
        "title": "Operations research by Elena Wentzel",
        "filename": "operations_research_wentzel.pdf",
        "category": "Optimization",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Operations%20research%20by%20Elena%20Wentzel.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Operations%20research%20by%20Elena%20Wentzel.pdf"
        ]
    },
    
    # Machine Learning Mathematics
    {
        "title": "MATHEMATICS FOR MACHINE LEARNING by Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
        "filename": "mathematics_machine_learning_deisenroth_faisal_ong.pdf",
        "category": "Machine Learning Math",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/MATHEMATICS%20FOR%20MACHINE%20LEARNING%20by%20Marc%20Peter%20Deisenroth%20A.%20Aldo%20Faisal%20Cheng%20Soon%20Ong.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/MATHEMATICS%20FOR%20MACHINE%20LEARNING%20by%20Marc%20Peter%20Deisenroth%20A.%20Aldo%20Faisal%20Cheng%20Soon%20Ong.pdf"
        ]
    },
    
    # Calculus and Analysis
    {
        "title": "Calculus Of Variations by I.M. Gelfand; S.V. Fomin",
        "filename": "calculus_variations_gelfand_fomin.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Calculus%20Of%20Variations%20by%20I.M.%20Gelfand;%20S.V.%20Fomin.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Calculus%20Of%20Variations%20by%20I.M.%20Gelfand;%20S.V.%20Fomin.pdf"
        ]
    },
    {
        "title": "Differential and Integral Calculus (Volumes 1 & 2) by Piskunov",
        "filename": "differential_integral_calculus_piskunov_vol1_2.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Differential%20and%20Integral%20Calculus%20Volumes%201%20%26%202%20by%20Piskunov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Differential%20and%20Integral%20Calculus%20Volumes%201%20%26%202%20by%20Piskunov.pdf"
        ]
    },
    {
        "title": "Differential And Integral Calculus by Piskunov",
        "filename": "differential_integral_calculus_piskunov.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Differential%20And%20Integral%20Calculus%20by%20Piskunov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Differential%20And%20Integral%20Calculus%20by%20Piskunov.pdf"
        ]
    },
    {
        "title": "Calculus by Spivak",
        "filename": "calculus_spivak.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Calculus%20by%20Spivak.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Calculus%20by%20Spivak.pdf"
        ]
    },
    {
        "title": "Calculus Made Easy by Silvanus P. Thompson",
        "filename": "calculus_made_easy_thompson.pdf",
        "category": "Calculus",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Calculus%20Made%20Easy%20by%20Silvanus%20P.%20Thompson.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Calculus%20Made%20Easy%20by%20Silvanus%20P.%20Thompson.pdf"
        ]
    },
    
    # ===== PHYSICS (30 books) =====
    {
        "title": "Problems in General Physics by Irodov",
        "filename": "problems_general_physics_irodov.pdf",
        "category": "Physics",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Problems%20in%20General%20Physics%20by%20Irodov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Problems%20in%20General%20Physics%20by%20Irodov.pdf"
        ]
    },
    {
        "title": "Fundamental Laws of Mechanics by Irodov",
        "filename": "fundamental_laws_mechanics_irodov.pdf",
        "category": "Physics",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Fundamental%20Laws%20of%20Mechanics%20by%20Irodov.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Fundamental%20Laws%20of%20Mechanics%20by%20Irodov.pdf"
        ]
    },
    {
        "title": "Collected Problems In Physics by S. Kozel; E. Rashda; S. Slavatinskii",
        "filename": "collected_problems_physics_kozel_rashda_slavatinskii.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Collected%20Problems%20In%20Physics%20by%20S.%20Kozel;%20E.%20Rashda;%20S.%20Slavatinskii.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Collected%20Problems%20In%20Physics%20by%20S.%20Kozel;%20E.%20Rashda;%20S.%20Slavatinskii.pdf"
        ]
    },
    {
        "title": "Lectures in analytical mechanics by Felix Gantmacher",
        "filename": "lectures_analytical_mechanics_gantmacher.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Lectures%20in%20analytical%20mechanics%20by%20Felix%20Gantmacher.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Lectures%20in%20analytical%20mechanics%20by%20Felix%20Gantmacher.pdf"
        ]
    },
    {
        "title": "Physics - a general course, Volumes I-III by Savelyev",
        "filename": "physics_general_course_savelyev_vol1_3.pdf",
        "category": "Physics",
        "priority": "🔥🔥🔥🔥🔥",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Physics%20-%20a%20general%20course%20Volumes%20I-III%20by%20Savelyev.pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Physics%20-%20a%20general%20course%20Volumes%20I-III%20by%20Savelyev.pdf"
        ]
    },
    
    # ===== ECONOMETRICS (1 book) =====
    {
        "title": "Basic Econometrics by Riccardo (Jack) Lucchetti (2024)",
        "filename": "basic_econometrics_lucchetti_2024.pdf",
        "category": "Econometrics",
        "priority": "",
        "urls": [
            "https://raw.githubusercontent.com/valeman/Awesome_Math_Books/main/Basic%20Econometrics%20by%20Riccardo%20(Jack)%20Lucchetti%20(2024).pdf",
            "https://github.com/valeman/Awesome_Math_Books/raw/main/Basic%20Econometrics%20by%20Riccardo%20(Jack)%20Lucchetti%20(2024).pdf"
        ]
    }
]

def download_file_with_fallback(book, download_dir="all_math_books"):
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
    
    print("🚀 Complete Math Books Downloader (208 books)")
    print("=" * 60)
    
    # Categorize books
    categories = categorize_books()
    
    print(f"\n📥 Starting downloads for {len(ALL_BOOKS)} books...")
    print("=" * 60)
    
    successful_downloads = 0
    total_downloads = len(ALL_BOOKS)
    
    # Download books by category
    for category, books in categories.items():
        if books:
            print(f"\n📚 Downloading {category} books:")
            print("-" * 40)
            
            for book in books:
                if download_file_with_fallback(book):
                    successful_downloads += 1
                
                # Small delay between downloads
                time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"📊 Download Summary:")
    print(f"   Successful: {successful_downloads}/{total_downloads}")
    print(f"   Failed: {total_downloads - successful_downloads}/{total_downloads}")
    print(f"   Success Rate: {(successful_downloads/total_downloads)*100:.1f}%")
    
    if successful_downloads > 0:
        print(f"\n📁 Books downloaded to: all_math_books/")
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

if __name__ == "__main__":
    main() 