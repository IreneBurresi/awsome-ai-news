#!/bin/bash
# Script per pulizia completa prima dei test

echo "🧹 Cleaning cache and generated files..."

# Remove cache
rm -rf cache

# Remove generated files
rm -rf news/*.yaml
rm -rf archive
rm -f feed.xml weekly.xml README.md

# Remove Python bytecode
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove pytest cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

echo "✅ Cleanup complete!"
