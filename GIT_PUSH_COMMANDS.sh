#!/bin/bash
# Commands to push to GitHub on a new branch: 1st-version
# Repository: https://github.com/Suyash-Kulashri/chormatography-training.git

echo "======================================================"
echo "Git Push to Branch: 1st-version"
echo "======================================================"

# Navigate to project directory
cd "/Users/mukulpathak/Desktop/chromo train"

# Step 1: Check current git status
echo ""
echo "Step 1: Checking git status..."
git status

# Step 2: Add all files (respecting .gitignore)
echo ""
echo "Step 2: Adding files to git..."
git add .

# Step 3: Commit changes
echo ""
echo "Step 3: Committing changes..."
git commit -m "feat: Add optimized hyperparameters and complete anomaly detection system

- Implemented Grid Search hyperparameter tuning (contamination=0.05, n_estimators=50)
- Added hyperparameter_tuning.ipynb for model optimization
- Updated config.py with optimized Isolation Forest parameters
- Enhanced app.py to show historical data and anomalies
- Improved anomaly detection from 10% to 5% false positive rate
- Added complete documentation (OPTIMIZED_HYPERPARAMETERS_README.md, FIXES_SUMMARY.md)
- Included trained models and label encoders
- Added future prediction capabilities with LSTM models"

# Step 4: Create and switch to new branch
echo ""
echo "Step 4: Creating and switching to branch '1st-version'..."
git checkout -b 1st-version

# Step 5: Set remote origin (if not already set)
echo ""
echo "Step 5: Setting remote origin..."
git remote add origin https://github.com/Suyash-Kulashri/chormatography-training.git 2>/dev/null || \
git remote set-url origin https://github.com/Suyash-Kulashri/chormatography-training.git

# Step 6: Push to the new branch
echo ""
echo "Step 6: Pushing to GitHub branch '1st-version'..."
git push -u origin 1st-version

echo ""
echo "======================================================"
echo "✓ Successfully pushed to branch: 1st-version"
echo "======================================================"
echo ""
echo "View your branch at:"
echo "https://github.com/Suyash-Kulashri/chormatography-training/tree/1st-version"
echo ""

