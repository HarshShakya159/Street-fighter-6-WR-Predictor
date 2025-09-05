# Street-fighter-6-WR-Predictor
A comprehensive machine learning project that analyzes and predicts Street Fighter 6 player win rates using core combat mechanics. Utilizes various ML Toolkits, and concepts: Supervised Learning, Regression Analysis, and Model Comparisons

# Supervised Learning
Regression Analysis: Implemented multiple regression algorithms to predict continuous win rate values
Model Comparison: Systematic evaluation of Linear Regression vs Random Forest performance
Cross-Validation: Used train-test split methodology

# Data Preprocessing & Feature Engineering
Feature Scaling: Applied StandardScaler for normalization of input features
Synthetic Data Generation: Created realistic training dataset with controlled statistical properties
Feature Selection: Identified 6 key performance indicators as predictive features
    - And based on my analysis, Most amount of PERFECT PARRIES lead to the most wins as seen in Professional/Pro-League Street        Fighter 6

  # Features Analyzed
  - Drive Impacts - Offensive pressure mechanic
  - Perfect Parries - Defensive timing skill
  - OD DPs - Special move execution
  - Level 1 Supers - Basic combo execution (10-15% HP Damage)
  - Level 2 Supers - Intermediate combos (25-30% HP Damage)
  - Level 3 Supers - Advanced combos (35-40% HP Damage)


# STATISTICAL ANALYSIS
- Normal distribution of win rates with realistic variance
- Strong positive correlation between technical execution and win rate

# TRAINING SET DATA AND FEATURES
Dynamic Player Generation: Random stat generation for continuous testing
Skill Assessment System: 6-tier classification from "Uninstall" to "Punk Level" (the best SF6 Player on the planet currently)
Real-time Predictions: Instant win rate calculation for new player profiles (being 3 new players to test against in my code)  
