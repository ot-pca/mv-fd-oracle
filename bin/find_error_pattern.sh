# Define the shared output file and error log
OUTPUT_FILE=./output/mv_error_patterns_hqc128.csv
MVORFD=mv
ERROR_LOG=./logs/error_log.txt

# Clear the error log file
> "$ERROR_LOG"

trap 'kill 0' SIGINT

# Function to run iterations independently for each core
run_task() {
    core_id=$1  # Get core id (just for tracking)
    
    # Loop x times independently
    for iteration in {1..2}; do
        echo "Core $core_id: Starting iteration $iteration..."

        # Redirect only stderr to the error log
        python ./src/find_error_pattern.py hqc128 "$MVORFD" "$OUTPUT_FILE" 2>> "$ERROR_LOG"

        status=$?
        if [ $status -ne 0 ]; then
            echo "Core $core_id: Error occurred in iteration $iteration. Check the error log: $ERROR_LOG"
            exit 1
        fi

        echo "Core $core_id: Iteration $iteration completed successfully."
    done
}

# Run tasks for 3 cores independently in parallel
run_task 1 &  # Core 1
pid1=$!

run_task 2 &  # Core 2
pid2=$!

run_task 3 &  # Core 3
pid3=$!

# Wait for all background jobs to finish
wait $pid1
wait $pid2
wait $pid3

echo "All tasks completed successfully."
