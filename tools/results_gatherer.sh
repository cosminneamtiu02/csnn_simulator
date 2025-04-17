#!/bin/bash

# Check if input directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_directory> [output_directory]"
    echo "  input_directory: Directory containing .txt files to process"
    echo "  output_directory: Directory where to save results.csv (default: current directory)"
    exit 1
fi

# Directory where log files are stored
input_dir="$1"

# Output directory (use provided directory or current directory)
output_dir="${2:-.}"

# Fixed filename
output_filename="results.csv"

# Full path to output file
output_csv="${output_dir}/${output_filename}"

# Temporary file for results before sorting
temp_csv="${output_dir}/temp_results.csv"

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Directory $input_dir does not exist."
    exit 1
fi

# Path to the data extractor script
extractor_script="/home/cosmin/proiecte/csnn-simulator/tools/ck+_3d_data_extractor.sh"

# Check if extractor script exists and is executable
if [ ! -x "$extractor_script" ]; then
    echo "Error: Extractor script $extractor_script not found or not executable."
    echo "Make sure to run: chmod +x $extractor_script"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory $output_dir"
        exit 1
    fi
fi

# Create CSV header
echo "id,seed,fold,epochs,width,height,depth,temporal_pooling,spatial_pooling,classification_rate" > "$output_csv"

# Create temporary file with header for storing results before sorting
echo "id,seed,fold,epochs,width,height,depth,temporal_pooling,spatial_pooling,classification_rate" > "$temp_csv"

# Counter for processed files
processed=0

# Find all .txt files and process them
for file_path in "$input_dir"/*.txt; do
    # Check if any files were found (if no matches, the pattern itself is returned)
    if [ ! -f "$file_path" ]; then
        echo "No .txt files found in $input_dir"
        break
    fi
    
    # Get just the filename (not the path)
    filename=$(basename "$file_path")
    
    # Process the file using the extractor script
    echo "Processing $filename..."
    
    # Run the extractor script and capture its output
    result=$("$extractor_script" "$input_dir" "$filename")
    
    # Check if extraction was successful
    if [ $? -eq 0 ] && [ ! -z "$result" ]; then
        # Replace spaces with commas for CSV format and append to temporary file
        echo "$result" | tr ' ' ',' >> "$temp_csv"
        ((processed++))
    else
        echo "Warning: Failed to process $filename, skipping"
    fi
done

echo "Processing complete. $processed files were analyzed."

# Only proceed with sorting if we have results
if [ $processed -gt 0 ]; then
    # Keep the header line
    head -1 "$temp_csv" > "$output_csv"
    
    # Sort the rest of the lines (skip header) by id, seed, and fold as integers
    tail -n +2 "$temp_csv" | sort -t, -k1,1n -k2,2n -k3,3n >> "$output_csv"
    
    # Remove temporary file
    rm "$temp_csv"
    
    echo "Results sorted by id, seed, and fold and saved to $output_csv"
else
    # Just copy the header if no results were found
    cp "$temp_csv" "$output_csv"
    rm "$temp_csv"
    echo "No results were found. Only header saved to $output_csv"
fi
