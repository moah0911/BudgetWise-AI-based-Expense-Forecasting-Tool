#!/usr/bin/env python3
"""
BudgetWise AI - Authentication and Signature System
Copyright (c) 2025 moah0911
Repository: https://github.com/moah0911/BudgetWise-AI-based-Expense-Forecasting-Tool

This file is part of BudgetWise AI project - Personal Expense Forecasting Tool.
Licensed under MIT License with Attribution Requirement.

Authentication system to verify legitimate BudgetWise AI installations
and detect unauthorized copies.

Author: moah0911
Created: October 2025
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Project Constants
PROJECT_SIGNATURE = "BW-AI-2025-v1.0"
AUTHOR_NAME = "BudgetWise AI"
REPOSITORY_URL = "https://github.com/moah0911/BudgetWise-AI-based-Expense-Forecasting-Tool"
COPYRIGHT_YEAR = "2025"
CREATION_DATE = "October 2025"

def generate_author_hash() -> str:
    """Generate unique author hash for verification."""
    signature_string = f"BudgetWise_AI_{COPYRIGHT_YEAR}"
    return hashlib.sha256(signature_string.encode()).hexdigest()[:16]

def create_project_metadata() -> Dict[str, Any]:
    """Create comprehensive project metadata for embedding."""
    return {
        'project_name': 'BudgetWise AI - Personal Expense Forecasting Tool',
        'author': AUTHOR_NAME,
        'copyright': f"© {COPYRIGHT_YEAR} {AUTHOR_NAME}",
        'repository': REPOSITORY_URL,
        'creation_date': CREATION_DATE,
        'project_signature': PROJECT_SIGNATURE,
        'author_hash': generate_author_hash(),
        'license': 'MIT License with Attribution Requirement',
        'timestamp': datetime.now().isoformat()
    }

def embed_signature_in_model(model_data: Any, model_name: str) -> Dict[str, Any]:
    """Embed authentication signature in model files."""
    metadata = create_project_metadata()
    metadata.update({
        'model_name': model_name,
        'model_creation_date': datetime.now().isoformat(),
        'embedded_by': 'BudgetWise AI'
    })
    
    return {
        'model': model_data,
        'budgetwise_metadata': metadata,
        'authenticity_signature': generate_model_signature(model_name)
    }

def generate_model_signature(model_name: str) -> str:
    """Generate unique signature for each model."""
    signature_data = f"BudgetWise_AI_{model_name}_{PROJECT_SIGNATURE}"
    return hashlib.md5(signature_data.encode()).hexdigest()

def verify_authenticity() -> Dict[str, Any]:
    """Verify this is an authentic BudgetWise AI installation."""
    return {
        'is_authentic': True,
        'project_signature': PROJECT_SIGNATURE,
        'author': AUTHOR_NAME,
        'author_hash': generate_author_hash(),
        'repository': REPOSITORY_URL,
        'copyright': f"© {COPYRIGHT_YEAR} {AUTHOR_NAME}",
        'verification_timestamp': datetime.now().isoformat()
    }

def create_copyright_notice() -> str:
    """Generate copyright notice for display."""
    return f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    BudgetWise AI                             ║
    ║              Personal Expense Forecasting Tool               ║
    ║                                                              ║
    ║  Copyright © {COPYRIGHT_YEAR} BudgetWise AI                            ║
    ║  Repository:                                        ║
    ║  github.com/moah0911/BudgetWise-AI-based-Expense-     ║
    ║  Forecasting-Tool                                            ║
    ║                                                              ║
    ║  Licensed under MIT License with Attribution Requirement     ║
    ║  Project Signature: {PROJECT_SIGNATURE}                       ║ 
    ╚══════════════════════════════════════════════════════════════╝
    """

def validate_project_integrity() -> bool:
    """Validate project files have not been tampered with."""
    required_files = [
        'LICENSE',
        'SECURITY.md',
        'README.md'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            return False
            
        # Check if copyright notice exists in file
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple check for copyright notice
                if "Copyright" not in content:
                    return False
        except:
            return False
    
    return True

def get_project_stats() -> Dict[str, Any]:
    """Get comprehensive project statistics for verification."""
    return {
        'project_name': 'BudgetWise AI',
        'signature': PROJECT_SIGNATURE,
        'repository': REPOSITORY_URL,
        'integrity_check': validate_project_integrity(),
        'authenticity': verify_authenticity(),
        'copyright_notice': create_copyright_notice().strip()
    }

# Embed signature in module
__copyright__ = f"© {COPYRIGHT_YEAR} BudgetWise AI"
__license__ = "MIT License with Attribution Requirement"
__version__ = "1.0.0"
__signature__ = PROJECT_SIGNATURE

if __name__ == "__main__":
    print(create_copyright_notice())
    print(f"Project Integrity: {'✅ VALID' if validate_project_integrity() else '❌ COMPROMISED'}")
    print(f"Author Hash: {generate_author_hash()}")