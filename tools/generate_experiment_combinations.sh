#!/bin/bash
# filepath: /home/cosmin/proiecte/csnn-simulator/tools/generate_experiment_combinations.sh

# Define parameter sets
height_width=(3 5 7)
depth=(3 4 5)
temp_pooling=(2 3 4)
seed=(6 7 23 24 30)

# Output file path - you can modify this variable to change the output location
output_file="runs.txt"

# Clear the output file if it exists
> "$output_file"

# Generate all combinations
counter=0
for hw in "${height_width[@]}"; do
    for d in "${depth[@]}"; do
        for tp in "${temp_pooling[@]}"; do
            for s in "${seed[@]}"; do
                # Generate the command with the current parameter values
                command="./Video_3d_CK_Plus_experiment $hw $hw $d $tp 800 $s"
                echo "$command" >> "$output_file"
                ((counter++))
            done
        done
    done
done