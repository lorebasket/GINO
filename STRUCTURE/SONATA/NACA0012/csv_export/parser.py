import numpy as np

def parse_sectional_matrix_csv(file_path):
    matrices = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect header: "section in r/R,0.0"
        if line.startswith("section in r/R"):
            section = line.split(',')[1].strip()
            i += 1  # move to next line

            # Read next 6 lines as a 6x6 matrix
            matrix_lines = lines[i:i+6]
            matrix = []
            for l in matrix_lines:
                row = [float(x) for x in l.strip().split(',') if x]
                matrix.append(row)
            matrices[section] = np.array(matrix)
            i += 6  # skip past this matrix
        else:
            i += 1  # skip unrelated lines

    return matrices
