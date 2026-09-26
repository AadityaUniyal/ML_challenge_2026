import os
import csv
import sys

def read_tsv(path):
    """Reads a TSV file with explicit tab separation and UTF-8 encoding."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)
    return header, rows

def stream_tsv_rows(path):
    """Generator yielding rows from a TSV file with header skipped."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if row:
                yield row
