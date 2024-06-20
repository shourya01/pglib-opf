import re

def extract_numbers_to_latex(input_file, output_file):
    pattern = re.compile(r'(\d+\.\d{5})')
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            matches = pattern.findall(line)
            if len(matches) >= 4:
                matches = matches[:4]
                outfile.write(f" & {matches[0]}\\% & {matches[1]}\\% & {matches[2]}\\% & {matches[3]}\\% \\\\\n")
# Usage
extract_numbers_to_latex('perf.txt', 'output.txt')