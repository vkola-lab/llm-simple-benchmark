
- The script `run_benchmark.sh` can either be run directly in bash (`$ ./run_benchmarks.sh`), or submitted to SCC (`$ qsub run_benchmarks.sh`). 
- We assume that the user is logged in to huggingface, we don't explicitly require a token.
- Each benchmark must be in its own directory inside the `benchmarks` directory, must be made of newline-delimited JSON files (a.k.a. JSONL), one problem per line.
