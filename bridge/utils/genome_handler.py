"""
BRIDGE Genome Data Handler
Manages genome sequences, annotations, and cross-species mappings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyfaidx
import pyBigWig
import pandas as pd
import numpy as np
from intervaltree import IntervalTree, Interval
import requests
import gzip
import logging
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class OrthologMapping:
    """Orthologous gene mapping between species"""
    source_gene_id: str
    source_species: str
    target_gene_id: str
    target_species: str
    confidence: float
    synteny_score: float

class GenomeDataHandler:
    """
    Central handler for all genome data operations
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.genomes = {}
        self.annotations = {}
        self.conservation_tracks = {}
        self.ortholog_mappings = {}
        
        # UCSC API endpoints
        self.ucsc_api = "https://api.genome.ucsc.edu"
        
        # Load available genomes
        self._load_genomes()
        
    def _load_genomes(self):
        """Load available genome files"""
        genome_dir = self.data_dir / "genomes"
        if genome_dir.exists():
            for fasta_file in genome_dir.glob("*.fa"):
                genome_name = fasta_file.stem
                try:
                    self.genomes[genome_name] = pyfaidx.Fasta(str(fasta_file))
                    logging.info(f"Loaded genome: {genome_name}")
                except Exception as e:
                    logging.error(f"Failed to load genome {genome_name}: {e}")
                    
    def fetch_sequence(self, chromosome: str, start: int, end: int, 
                      genome: str = "hg38") -> str:
        """
        Fetch genomic sequence for a region
        
        Args:
            chromosome: Chromosome name (e.g., 'chr1')
            start: Start position (0-based)
            end: End position
            genome: Genome assembly (default: hg38)
            
        Returns:
            DNA sequence string
        """
        if genome not in self.genomes:
            # Try to load from file
            genome_path = self.data_dir / "genomes" / f"{genome}.fa"
            if genome_path.exists():
                self.genomes[genome] = pyfaidx.Fasta(str(genome_path))
            else:
                # Fallback to UCSC API
                return self._fetch_from_ucsc(chromosome, start, end, genome)
                
        try:
            # Fetch from local file
            seq = self.genomes[genome][chromosome][start:end].seq.upper()
            return seq
        except KeyError:
            logging.warning(f"Chromosome {chromosome} not found in {genome}")
            return self._fetch_from_ucsc(chromosome, start, end, genome)
            
    def _fetch_from_ucsc(self, chromosome: str, start: int, end: int, 
                        genome: str) -> str:
        """Fetch sequence from UCSC API"""
        url = f"{self.ucsc_api}/getData/sequence"
        params = {
            "genome": genome,
            "chrom": chromosome,
            "start": start,
            "end": end
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("dna", "")
        except Exception as e:
            logging.error(f"Failed to fetch from UCSC: {e}")
            # Return Ns as placeholder
            return "N" * (end - start)
            
    @lru_cache(maxsize=1000)
    def get_conservation_score(self, chromosome: str, start: int, end: int,
                             genome: str = "hg38", 
                             conservation_type: str = "phyloP") -> np.ndarray:
        """
        Get conservation scores for a region
        
        Args:
            chromosome: Chromosome name
            start: Start position
            end: End position
            genome: Genome assembly
            conservation_type: 'phyloP' or 'phastCons'
            
        Returns:
            Array of conservation scores
        """
        track_key = f"{genome}_{conservation_type}"
        
        if track_key not in self.conservation_tracks:
            # Load conservation track
            track_file = self.data_dir / "conservation" / f"{genome}.{conservation_type}.bw"
            if track_file.exists():
                self.conservation_tracks[track_key] = pyBigWig.open(str(track_file))
            else:
                logging.warning(f"Conservation track not found: {track_file}")
                # Return neutral scores
                return np.zeros(end - start)
                
        try:
            bw = self.conservation_tracks[track_key]
            scores = bw.values(chromosome, start, end)
            # Replace None values with 0
            scores = np.array([s if s is not None else 0 for s in scores])
            return scores
        except Exception as e:
            logging.error(f"Error fetching conservation scores: {e}")
            return np.zeros(end - start)
            
    def find_orthologs(self, gene_id: str, source_species: str, 
                      target_species: str) -> List[OrthologMapping]:
        """
        Find orthologous genes between species
        
        Args:
            gene_id: Gene ID in source species
            source_species: Source species name
            target_species: Target species name
            
        Returns:
            List of ortholog mappings
        """
        mapping_key = f"{source_species}_{target_species}"
        
        if mapping_key not in self.ortholog_mappings:
            # Load ortholog mappings
            self._load_ortholog_mappings(source_species, target_species)
            
        if mapping_key in self.ortholog_mappings:
            # Look up gene
            orthologs = []
            for mapping in self.ortholog_mappings[mapping_key]:
                if mapping.source_gene_id == gene_id:
                    orthologs.append(mapping)
                    
            return orthologs
        else:
            return []
            
    def _load_ortholog_mappings(self, source: str, target: str):
        """Load ortholog mappings from file or database"""
        # Check for local file
        mapping_file = self.data_dir / "orthologs" / f"{source}_to_{target}.tsv"
        
        if mapping_file.exists():
            df = pd.read_csv(mapping_file, sep='\t')
            mappings = []
            
            for _, row in df.iterrows():
                mapping = OrthologMapping(
                    source_gene_id=row['source_gene_id'],
                    source_species=source,
                    target_gene_id=row['target_gene_id'],
                    target_species=target,
                    confidence=row.get('confidence', 1.0),
                    synteny_score=row.get('synteny_score', 0.0)
                )
                mappings.append(mapping)
                
            self.ortholog_mappings[f"{source}_{target}"] = mappings
        else:
            # Could query Ensembl Compara or other databases
            logging.warning(f"No ortholog mapping found for {source} to {target}")
            self.ortholog_mappings[f"{source}_{target}"] = []
            
    def get_orthologous_sequence(self, chromosome: str, start: int, end: int,
                                source_species: str, target_species: str,
                                extend_flanks: bool = True) -> Optional[str]:
        """
        Get orthologous sequence from another species
        
        Args:
            chromosome: Chromosome in source species
            start: Start position in source species
            end: End position in source species
            source_species: Source species genome
            target_species: Target species genome
            extend_flanks: Whether to extend flanking regions
            
        Returns:
            Orthologous sequence or None if not found
        """
        # First, find genes in the region
        genes_in_region = self.find_genes_in_region(
            chromosome, start, end, source_species
        )
        
        if not genes_in_region:
            logging.warning(f"No genes found in {chromosome}:{start}-{end}")
            return None
            
        # Find orthologs for the genes
        for gene in genes_in_region:
            orthologs = self.find_orthologs(
                gene['gene_id'], source_species, target_species
            )
            
            if orthologs:
                # Use best ortholog
                best_ortholog = max(orthologs, key=lambda x: x.confidence)
                
                # Get ortholog coordinates
                target_gene = self.get_gene_info(
                    best_ortholog.target_gene_id, target_species
                )
                
                if target_gene:
                    # Calculate relative position in gene
                    gene_start = gene['start']
                    gene_end = gene['end']
                    
                    rel_start = (start - gene_start) / (gene_end - gene_start)
                    rel_end = (end - gene_start) / (gene_end - gene_start)
                    
                    # Map to target gene
                    target_length = target_gene['end'] - target_gene['start']
                    target_start = int(target_gene['start'] + rel_start * target_length)
                    target_end = int(target_gene['start'] + rel_end * target_length)
                    
                    if extend_flanks:
                        # Add 20% flanking regions
                        flank = int(0.2 * (target_end - target_start))
                        target_start -= flank
                        target_end += flank
                        
                    # Fetch sequence
                    return self.fetch_sequence(
                        target_gene['chromosome'],
                        target_start,
                        target_end,
                        target_species
                    )
                    
        return None
        
    def find_genes_in_region(self, chromosome: str, start: int, end: int,
                            genome: str = "hg38") -> List[Dict]:
        """Find genes in a genomic region"""
        # This would integrate with GeneAnnotationManager
        # For now, return placeholder
        return []
        
    def get_gene_info(self, gene_id: str, species: str) -> Optional[Dict]:
        """Get gene information"""
        # This would query gene database
        # For now, return placeholder
        return None
        
    def get_species_genome_map(self) -> Dict[str, str]:
        """Map species names to genome assemblies"""
        return {
            "human": "hg38",
            "mouse": "mm10",
            "rat": "rn6",
            "zebrafish": "danRer11",
            "chicken": "galGal6",
            "fly": "dm6",
            "worm": "ce11"
        }
        
    def download_genome(self, genome: str, species: str = None):
        """Download genome from UCSC"""
        genome_dir = self.data_dir / "genomes"
        genome_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = genome_dir / f"{genome}.fa"
        if output_file.exists():
            logging.info(f"Genome {genome} already downloaded")
            return
            
        url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/bigZips/{genome}.fa.gz"
        
        logging.info(f"Downloading {genome} from {url}")
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        gz_file = output_file.with_suffix('.fa.gz')
        
        with open(gz_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Decompress
        logging.info(f"Decompressing {genome}")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())
                
        gz_file.unlink()  # Remove compressed file
        
        # Index with pyfaidx
        logging.info(f"Indexing {genome}")
        pyfaidx.Faidx(str(output_file))
        
        # Load into memory
        self.genomes[genome] = pyfaidx.Fasta(str(output_file))
        
        logging.info(f"Successfully loaded {genome}")

# Singleton instance
_genome_handler = None

def get_genome_handler(data_dir: str = "data") -> GenomeDataHandler:
    """Get or create genome handler instance"""
    global _genome_handler
    if _genome_handler is None:
        _genome_handler = GenomeDataHandler(data_dir)
    return _genome_handler
