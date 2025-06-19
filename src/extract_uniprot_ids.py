#!/usr/bin/env python3
"""
UniProt ID Extraction Script using MMseqs2 easy-search

This script extracts UniProt IDs from protein sequences using MMseqs2 easy-search
and appends them to JSON files in the new_embeddings folder.

Usage:
    python extract_uniprot_ids.py
"""

import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniRef50Searcher:
    def __init__(self, db_path="/media/shishir/Backup/uniref50_db/uniref50"):
        """Initialize the UniRef50 searcher using mmseqs easy-search."""
        self.db_path = db_path
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if MMseqs2 and database are available."""
        try:
            subprocess.run(["mmseqs", "--help"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("MMseqs2 is not installed or not working")
            sys.exit(1)
            
        if not (os.path.exists(f"{self.db_path}") and os.path.exists(f"{self.db_path}.dbtype")):
            logger.error(f"UniRef50 database not found: {self.db_path}")
            sys.exit(1)
        
    def create_fasta_from_json(self, json_data: List[Dict[str, Any]], output_file: str) -> bool:
        """Create a FASTA file from JSON data."""
        sequences_written = 0
        with open(output_file, 'w') as f:
            for entry in json_data:
                entry_id = entry.get('id') or entry.get('ID')
                sequence = entry.get('sequence') or entry.get('seq')
                
                if entry_id and sequence and all(c in 'ACDEFGHIKLMNPQRSTVWYXBZJUO*-' for c in sequence.upper()):
                    f.write(f">{entry_id}\n{sequence}\n")
                    sequences_written += 1
        
        logger.info(f"Created FASTA file with {sequences_written} sequences")
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0
    
    def search_sequences_with_fasta(self, query_fasta: str) -> Dict[str, str]:
        """Search for UniProt IDs using mmseqs easy-search."""
        temp_base = "/tmp/temp_mmseqs"
        os.makedirs(temp_base, exist_ok=True)
        
        with tempfile.TemporaryDirectory(dir=temp_base) as temp_dir:
            result_file = os.path.join(temp_dir, "results.m8")
            
            cmd = [
                "mmseqs", "easy-search",
                query_fasta,
                "/media/shishir/Backup/uniref50_db/uniref50.fasta",
                result_file,
                temp_dir,
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return self.parse_results(result_file)
            except subprocess.CalledProcessError as e:
                logger.error(f"mmseqs easy-search failed: {e}")
                return {}
    
    def parse_results(self, results_file: str) -> Dict[str, str]:
        """Parse mmseqs easy-search results and extract best UniProt IDs."""
        if not os.path.exists(results_file):
            return {}
        
        try:
            df = pd.read_csv(results_file, sep='\t', header=None, names=[
                'query', 'target', 'pident', 'alnlen', 'mismatch', 'gapopen',
                'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'
            ])
            
            if df.empty:
                return {}
            
            df_filtered = df[df['pident'] >= 99.0]
            results = {}
            
            for query_id in df_filtered['query'].unique():
                query_hits = df_filtered[df_filtered['query'] == query_id]
                best_hit = query_hits.sort_values(['pident', 'evalue'], ascending=[False, True]).iloc[0]
                
                uniprot_id = self.extract_uniprot_id(best_hit['target'])
                if uniprot_id:
                    results[query_id] = uniprot_id
            
            logger.info(f"Found matches for {len(results)} sequences")
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            return {}
    
    def extract_uniprot_id(self, target: str) -> str:
        """Extract UniProt ID from UniRef50 target identifier."""
        if target.startswith("UniRef50_"):
            uniprot_id = target.replace("UniRef50_", "")
            if len(uniprot_id) >= 6 and uniprot_id.replace("_", "").isalnum():
                return uniprot_id
        return None

class JSONProcessor:
    def __init__(self, json_files: List[str]):
        """Initialize JSON processor."""
        self.json_files = json_files
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
    
    def save_json_data(self, data: List[Dict[str, Any]], file_path: str):
        """Save JSON data to file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_with_uniprot_ids(self, data: List[Dict[str, Any]], uniprot_results: Dict[str, str]) -> List[Dict[str, Any]]:
        """Update JSON data with UniProt IDs."""
        for entry in data:
            entry_id = entry.get('id') or entry.get('ID')
            entry['UniProt_ID'] = uniprot_results.get(entry_id)
        return data

def main():
    """Main function to process JSON files and extract UniProt IDs."""
    json_files = ["new_embeddings/NMR_plddt.json"]
    
    for file_path in json_files:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
    
    searcher = UniRef50Searcher()
    processor = JSONProcessor(json_files)
    results_summary = {}
    
    for json_file in json_files:
        logger.info(f"Processing {json_file}...")
        
        data = processor.load_json_data(json_file)
        if not data:
            continue
        
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        fasta_file = f"{base_name}.fasta"
        
        if not searcher.create_fasta_from_json(data, fasta_file):
            logger.error(f"Failed to create FASTA file for {json_file}")
            continue
        
        uniprot_results = searcher.search_sequences_with_fasta(fasta_file)
        updated_data = processor.update_with_uniprot_ids(data, uniprot_results)
        
        backup_file = json_file + ".backup"
        if not Path(backup_file).exists():
            processor.save_json_data(data, backup_file)
        
        processor.save_json_data(updated_data, json_file)
        
        try:
            os.remove(fasta_file)
        except:
            pass
        
        results_summary[base_name] = {
            'total': len(data),
            'found': len(uniprot_results),
            'percentage': (len(uniprot_results) / len(data) * 100) if data else 0
        }
    
    logger.info("SUMMARY:")
    total_all = found_all = 0
    
    for dataset, stats in results_summary.items():
        logger.info(f"{dataset}: {stats['found']}/{stats['total']} ({stats['percentage']:.1f}%)")
        total_all += stats['total']
        found_all += stats['found']
    
    overall_percentage = (found_all / total_all * 100) if total_all > 0 else 0
    logger.info(f"Overall: {found_all}/{total_all} ({overall_percentage:.1f}%)")

if __name__ == "__main__":
    main() 