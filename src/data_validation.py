"""
Data Contract & Schema Validation Module.
Validates input files, entity ID prefixes, uniqueness, and completeness.
"""
import os
import csv

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def validate_source_file(path, expected_prefix):
    """Validates schema, encoding, and ID prefixes for a source TSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    ids = set()
    total_rows = 0
    missing_name = 0
    missing_addr = 0
    countries = set()

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if header != REQUIRED_COLUMNS:
            raise ValueError(f"Invalid header in {path}: {header}, expected {REQUIRED_COLUMNS}")

        for i, row in enumerate(reader, start=2):
            total_rows += 1
            if len(row) < 4:
                raise ValueError(f"Malformed row at line {i} in {path}: {row}")

            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if not eid.startswith(expected_prefix):
                raise ValueError(f"Invalid prefix '{eid}' in {path}, expected '{expected_prefix}'")
            if eid in ids:
                raise ValueError(f"Duplicate entity_id '{eid}' at line {i} in {path}")
            ids.add(eid)

            if not name.strip():
                missing_name += 1
            if not addr.strip():
                missing_addr += 1
            countries.add(country.strip())

    return {
        'total_rows': total_rows,
        'unique_ids': len(ids),
        'missing_name': missing_name,
        'missing_addr': missing_addr,
        'countries': list(countries)
    }
