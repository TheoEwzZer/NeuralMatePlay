#!/usr/bin/env python3
"""
Neural Network Diagnostic Suite for Chess Engine.

This is a wrapper script that calls the modular diagnostics package.
For the actual implementation, see the diagnostics/ directory.

Usage:
    python diagnose_network.py [checkpoint_dir] [options]

Options:
    -s, --model MODEL    Test a single specific model file
    -i, --iteration N    Test specific iteration
    -c, --compare        Compare all checkpoints
    -m, --models M1 M2   Compare two specific model files
    -t, --type TYPE      Checkpoint type: train or pretrain
    -q, --quick          Quick comparison (fewer tests)

Examples:
    python diagnose_network.py models                    # Test latest checkpoint
    python diagnose_network.py models -t pretrain       # Test pretrain checkpoints
    python diagnose_network.py models -c                # Compare all checkpoints
    python diagnose_network.py -s path/to/model.pt      # Test a specific model
    python diagnose_network.py -m model1.pt model2.pt   # Compare two models
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from diagnostics.main import main

if __name__ == "__main__":
    main()
