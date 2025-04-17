#!/bin/bash

# Check if both path and filename were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <path> <filename>"
    exit 1
fi

# Store the path and filename
path="$1"
filename="$2"

# Combine path and filename
full_path="${path}/${filename}"

# Check if file exists
if [ ! -f "$full_path" ]; then
    echo "Error: File $full_path does not exist."
    exit 1
fi

# Extract parameters from filename
# Extract ID (the longest numeric sequence in the filename)
id=$(echo "$filename" | grep -o '[0-9]\{9,\}' | head -1)

# Extract dimensions from pattern like 7x7x5
dimensions=$(echo "$filename" | grep -o '[0-9]\+x[0-9]\+x[0-9]\+')
width=$(echo "$dimensions" | cut -d'x' -f1)
height=$(echo "$dimensions" | cut -d'x' -f2)
depth=$(echo "$dimensions" | cut -d'x' -f3)

# Extract temporal pooling from tp<number>
temporal_pooling=$(echo "$filename" | grep -o 'tp[0-9]\+' | grep -o '[0-9]\+')

# Extract spatial pooling from sp<number>
spatial_pooling=$(echo "$filename" | grep -o 'sp[0-9]\+' | grep -o '[0-9]\+')

# Extract fold from fold<number>
fold=$(echo "$filename" | grep -o 'fold[0-9]\+' | grep -o '[0-9]\+')

# Extract epochs from epochs<number>
epochs=$(echo "$filename" | grep -o 'epochs[0-9]\+' | grep -o '[0-9]\+')

# Extract seed from seed<number>
seed=$(echo "$filename" | grep -o 'seed[0-9]\+' | grep -o '[0-9]\+')

# Extract classification rate from file content
# Modified to handle both integer and floating point percentages
classification_rate=$(grep "classification rate:" "$full_path" | grep -o '[0-9]\+\(\.[0-9]\+\)\?%' | tr -d '%')

# Check if we got a classification rate, if not, print debug info and fail
if [ -z "$classification_rate" ]; then
    echo "Error: Could not find classification rate in $filename" >&2
    echo "Debug info: Classification line from file:" >&2
    grep "classification rate:" "$full_path" >&2
    exit 1
fi

# Output all parameters in the specified order
echo "$id $seed $fold $epochs $width $height $depth $temporal_pooling $spatial_pooling $classification_rate"
