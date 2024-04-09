#!/bin/bash

# Directories to process
dirs=(
    "/tsukimi/datasets/rats-v2ce/train/dvs-rat-recording-official"
    "/tsukimi/datasets/rats-v2ce/train/Pilot-Data-20231127"
    "/tsukimi/datasets/rats-v2ce/train/testdata-12-18"
)

# Output directory
outdir="/tsukimi/datasets/Chiba/finetune_train"

# Error log file
error_log_file="./error_log.txt"

# Iterate over each directory
for dir in "${dirs[@]}"; do
    # Iterate over each .aedat4 file in the directory
    for file_path in "$dir"/*.aedat4; do

        if [[ $filename == .* ]]; then
            continue
        fi

        filename=$(basename "$file_path")
        echo "processing $filename"

        # Run the command
        if python ./event_chunk.py -i "$file_path" -o "$outdir"; then
            echo "$filename completed"
        else
            # Log the error
            echo "Error processing file $filename" >> "$error_log_file"
        fi
    done
done