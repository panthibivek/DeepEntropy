#!/usr/bin/env python3
"""
Simple script to convert softDis_plddt.json to FASTA format
"""

import json

def json_to_fasta(json_file, output_file):
    """Convert JSON file with sequences to FASTA format"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for entry in data:
            # Write FASTA header with ID
            f.write(f">{entry['id']}\n")
            # Write sequence
            f.write(f"{entry['sequence']}\n")
    
    print(f"Converted {len(data)} sequences to {output_file}")

if __name__ == "__main__":
    # Convert the softDis_plddt.json file to FASTA
    json_to_fasta("softDis_plddt.json", "softdis_sequences.fasta") 