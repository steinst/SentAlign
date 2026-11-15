import sys
import os

def extract_indices(bracket_str):
    """
    Strips the brackets and returns a list of integers.
    Returns an empty list if there are no numbers inside.
    """
    clean_str = bracket_str.strip('[]')
    if not clean_str:
        return []
    return [int(x) for x in clean_str.split(',')]

def main():
    # File paths based on your directory structure
    path_file = sys.argv[1] #"samrodun/output/candide.txt.path"
    fra_file = sys.argv[2] #"samrodun/fra/candide.txt"
    isl_file = sys.argv[3] #"samrodun/isl/candide.txt"
    out_file = sys.argv[4] #"samrodun/output/aligned_candide.tsv"

    # Read the text lines into memory and strip trailing newlines/whitespace
    with open(fra_file, 'r', encoding='utf-8') as f:
        fra_lines = [line.strip() for line in f]
        
    with open(isl_file, 'r', encoding='utf-8') as f:
        isl_lines = [line.strip() for line in f]

    # Open the path file for reading and the TSV file for writing
    with open(path_file, 'r', encoding='utf-8') as pf, \
         open(out_file, 'w', encoding='utf-8') as out:
        
        for line in pf:
            line = line.strip()
            if not line:
                continue
            
            # Split the line by colons: "[0]:[0,1]:0.55574024" -> ["[0]", "[0,1]", "0.55574024"]
            parts = line.split(':')
            if len(parts) != 3:
                continue # Skip any malformed lines
            
            fra_part, isl_part, score = parts
            
            # Parse the bracketed numbers into lists of integers
            fra_indices = extract_indices(fra_part)
            isl_indices = extract_indices(isl_part)
            
            # Fetch the corresponding text lines and join them with a single space
            # The inline if-statement protects against out-of-bound indexing just in case
            fra_text = " ".join([fra_lines[i] for i in fra_indices if i < len(fra_lines)])
            isl_text = " ".join([isl_lines[i] for i in isl_indices if i < len(isl_lines)])
            
            # Write the formatted tab-separated row to the output file
            out.write(f"{fra_text}\t{isl_text}\t{score}\n")

    print(f"Alignment complete. Output saved to: {out_file}")

if __name__ == "__main__":
    main()
