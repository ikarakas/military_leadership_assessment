#!/bin/bash

# Script to add license headers to all Python files in the project
# Usage: ./add_license_headers.sh

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LICENSE_FILE="$PROJECT_ROOT/LICENSE"

# Check if license file exists
if [ ! -f "$LICENSE_FILE" ]; then
    echo "Error: License file not found at $LICENSE_FILE"
    exit 1
fi

# Create a temporary file for the license header
TEMP_LICENSE=$(mktemp)

# Read the license file and format it as a Python docstring
echo '"""' > "$TEMP_LICENSE"
cat "$LICENSE_FILE" >> "$TEMP_LICENSE"
echo '"""' >> "$TEMP_LICENSE"

# Find all Python files in the project (excluding virtual environment)
find "$PROJECT_ROOT" -type f -name "*.py" -not -path "*/venv/*" -not -path "*/__pycache__/*" | while read -r file; do
    echo "Processing $file"
    
    # Create a temporary file for the modified content
    TEMP_FILE=$(mktemp)
    
    # Check if the file already has a license header
    if grep -q "Copyright (c) 2025 Ilker M. Karakas" "$file"; then
        echo "License header already exists in $file, skipping..."
        continue
    fi
    
    # Add the license header to the beginning of the file
    cat "$TEMP_LICENSE" > "$TEMP_FILE"
    cat "$file" >> "$TEMP_FILE"
    
    # Replace the original file with the modified content
    mv "$TEMP_FILE" "$file"
    
    echo "Added license header to $file"
done

# Clean up temporary files
rm "$TEMP_LICENSE"

echo "License header addition complete!" 