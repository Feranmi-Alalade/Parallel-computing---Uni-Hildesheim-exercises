#!/usr/bin/env python3
import sys

def main():
    current_index = None
    values = {'A': None, 'B': None}
    dot_product = 0.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            index, vector_id, value = line.split('\t')
            value = float(value)
            
            if index != current_index:
                # Calculate product for previous index if we have both values
                if current_index is not None and all(values.values()):
                    dot_product += values['A'] * values['B']
                current_index = index
                values = {'A': None, 'B': None}
            
            values[vector_id] = value
            
            # If we have both values for current index, multiply them
            if all(values.values()):
                dot_product += values['A'] * values['B']
                values = {'A': None, 'B': None}
                
        except Exception as e:
            sys.stderr.write(f"ERROR processing line '{line}': {str(e)}\n")
    
    # Output final dot product
    print(f"Dot Product: {dot_product}")

if __name__ == "__main__":
    main()