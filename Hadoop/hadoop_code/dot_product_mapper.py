#!/usr/bin/env python3
import sys

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            index, vector_id, value = line.split(" ")
            # Emit the index as key, (vector_id, value) as value
            print(f"{index}\t{vector_id}\t{value}")
        except ValueError:
            sys.stderr.write(f"ERROR: Invalid line format: {line}\n")

if __name__ == "__main__":
    main()