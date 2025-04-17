#!/bin/bash
# filepath: /home/cosmin/proiecte/csnn-simulator/tools/generate_experiment_combinations.sh

# Define parameter sets
height_width=(3 5 7)
depth=(2 3 4 5)  # Added depth 2 as mentioned in requirements
seed=(42 13 93 45 96 6 98 59 44)

# Output file path - you can modify this variable to change the output location
output_file="runs.txt"

# Clear the output file if it exists
> "$output_file"

# Generate all combinations
counter=0
for hw in "${height_width[@]}"; do
    for d in "${depth[@]}"; do
        # Set temp_pooling values based on depth
        if [ "$d" -eq 2 ]; then
            # For depth 2, FMap has depth 9, use temp_pooling 2,3,4,5
            temp_pooling_values=(2 3 4 5)
        elif [ "$d" -eq 3 ]; then
            # For depth 3, FMap has depth 8, use temp_pooling 2,3,4
            temp_pooling_values=(2 3 4)
        elif [ "$d" -eq 4 ]; then
            # For depth 4, FMap has depth 7, use temp_pooling 2,3
            temp_pooling_values=(2 3)
        elif [ "$d" -eq 5 ]; then
            # For depth 5, FMap has depth 6, use temp_pooling 2,3
            temp_pooling_values=(2 3)
        fi

        for tp in "${temp_pooling_values[@]}"; do
            for s in "${seed[@]}"; do
                # Generate the command with the current parameter values
                command="./Video_3d_CK_Plus_experiment $hw $hw $d $tp 800 $s"
                echo "$command" >> "$output_file"
                ((counter++))
            done
        done
    done
done

echo "Generated $counter experiment combinations in $output_file"