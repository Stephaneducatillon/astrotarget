#!/usr/bin/env python3
"""
Script d'entraînement autonome — raccourci vers models/train.py.

Usage depuis la racine f1_prediction/ :
    python scripts/train_model.py
    python scripts/train_model.py --years 2022 2023 2024 2025
    python scripts/train_model.py --mode qualifying
"""
import sys
import os

# Ajouter le parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import main

if __name__ == "__main__":
    main()
