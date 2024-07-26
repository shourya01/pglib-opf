import re

def extract_numbers_to_latex(input_file, output_file):
    pattern = re.compile(r'(\d+\.\d{5})')
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            matches = pattern.findall(line)
            matches = [float(m) for m in matches]
            if len(matches) == 5:
                outfile.write(f"& {matches[4]:.4f}s & {matches[0]:.2f}\\% & {matches[1]:.2f}\\% & {matches[2]:.2f}\\% & {matches[3]:.2f}\\% & \\\\\n")
                outfile.write(f"fn: {matches[3]:.6f}, positives: {(matches[0]+matches[3]):.2f}, negatives: {(matches[1]+matches[2]):.2f}\n")
# Usage
extract_numbers_to_latex('perf.txt', 'output.txt')