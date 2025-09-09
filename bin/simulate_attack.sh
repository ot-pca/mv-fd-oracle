SCHEME=$1
MVORFD=$2
RHO=$3
NUM_ERROR_PATTERNS=$4

# Run the Python script 
time python3 ./src/simulate_attack.py "$SCHEME" "$MVORFD" "$RHO" "$NUM_ERROR_PATTERNS"