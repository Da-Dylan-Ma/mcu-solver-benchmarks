import csv
from collections import defaultdict

csv_file = "logs.csv"

patterns = defaultdict(lambda: {'P': {'p': [], 'i': []}, 'A': {'p': [], 'i': []}})

line_count = 0
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        line_count += 1
        iteration = int(row['iteration'])
        mat = row['matrix_name']
        arr_type = row['array_type']
        value = int(row['value'])
        patterns[iteration][mat][arr_type].append(value)

print(f"Finished parsing {line_count} lines from {csv_file}.")
print(f"Number of iterations parsed: {len(patterns.keys())}")
print("Iterations found:", sorted(patterns.keys()))

def arrays_equal(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))

changes = []
if patterns:
    max_iter = max(patterns.keys())
    for it in range(max_iter):
        next_it = it + 1
        if next_it in patterns:
            for mat in ['P', 'A']:
                p_same = arrays_equal(patterns[it][mat]['p'], patterns[next_it][mat]['p'])
                i_same = arrays_equal(patterns[it][mat]['i'], patterns[next_it][mat]['i'])
                if not p_same or not i_same:
                    changes.append((it, next_it, mat, not p_same, not i_same))

if not changes:
    print("No changes in sparsity pattern detected across iterations.")
else:
    for c in changes:
        old_iter, new_iter, mat, p_changed, i_changed = c
        print(f"Sparsity pattern changed from iteration {old_iter} to {new_iter} in {mat}.")
        if p_changed:
            print("  p array changed.")
        if i_changed:
            print("  i array changed.")
