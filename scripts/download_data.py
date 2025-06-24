#!/usr/bin/env python3
"""
Download required data and models for BRIDGE
Includes genomic references, annotations, and pre-trained weights
"""

import os
import sys
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import hashlib
import subprocess
from typing import Dict, List, Optional
import argparse

# Data sources configuration
DATA_SOURCES = {
    "genomes": {
        "hg38": {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            "md5": "22e7e49c726e4fa349143e3f78b36de5",
            "size": "938M",
            "description": "Human reference genome (GRCh38/hg38)"
        },
        "mm10": {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
            "md5": "70166e616ba6a59e23dd9e7a07419665",
            "size": "830M",
            "description": "Mouse reference genome (mm10)"
        },
        "danRer11": {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz",
            "md5": "7e256a3e43492ee7e982d1f1499e7f6e",
            "size": "370M",
            "description": "Zebrafish reference genome (GRCz11)"
        }
    },
    "annotations": {
        "hg38_genes": {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.refGene.gtf.gz",
            "md5": "b5f95dd4b749f858f9d0c5f7f0e6e5a7",
            "size": "15M",
            "description": "Human gene annotations (RefSeq)"
        },
        "hg38_encode": {
            "url": "https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz",
            "size": "450K",
            "description": "ENCODE blacklist regions for hg38"
        },
        "hg38_cpg": {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
            "size": "2.8M",
            "description": "CpG islands for hg38"
        }
    },
    "regulatory": {
        "vista_enhancers": {
            "url": "https://enhancer.lbl.gov/cgi-bin/imagedb3.pl?page=1&form=ext_search&show=10000&search.result=yes&search.org=all&search.gene=all",
            "format": "custom",
            "size": "5M",
            "description": "VISTA validated enhancers"
        },
        "encode_cres": {
            "url": "https://api.wenglab.org/screen_v13/fdownloads/cCREs/V3/hg38-cCREs.bed",
            "size": "180M",
            "description": "ENCODE candidate cis-regulatory elements"
        },
        "fantom5_cage": {
            "url": "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz",
            "size": "45M",
            "description": "FANTOM5 CAGE peaks (promoters/enhancers)"
        }
    },
    "conservation": {
        "phylop100": {
            "url": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
            "size": "9.8G",
            "description": "100-way vertebrate conservation (phyloP)"
        },
        "phastcons100": {
            "url": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
            "size": "1.2G",
            "description": "100-way vertebrate conservation (phastCons)"
        }
    },
    "models": {
        "hyenadna_weights": {
            "url": "https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen/resolve/main/pytorch_model.bin",
            "size": "6.8G",
            "description": "HyenaDNA pre-trained weights"
        },
        "dnabert2_weights": {
            "url": "https://huggingface.co/zhihan1996/DNABERT-2-117M/resolve/main/pytorch_model.bin",
            "size": "440M",
            "description": "DNABERT-2 pre-trained weights"
        },
        "bridge_finetuned": {
            "url": "https://example.com/bridge_finetuned_v1.tar.gz",
            "size": "2.1G",
            "description": "BRIDGE fine-tuned weights (optional)",
            "optional": True
        }
    }
}

class DataDownloader:
    """Handle data downloads with progress tracking and verification"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_file(self, url: str, dest_path: Path, 
                     expected_size: Optional[str] = None,
                     expected_md5: Optional[str] = None) -> bool:
        """Download file with progress bar and verification"""
        
        # Check if already downloaded
        if dest_path.exists():
            if expected_md5 and self.verify_md5(dest_path, expected_md5):
                print(f"✓ {dest_path.name} already downloaded and verified")
                return True
            elif not expected_md5:
                print(f"✓ {dest_path.name} already exists")
                return True
                
        print(f"Downloading {dest_path.name}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            # Verify download
            if expected_md5:
                if self.verify_md5(dest_path, expected_md5):
                    print(f"✓ {dest_path.name} downloaded and verified")
                    return True
                else:
                    print(f"✗ MD5 mismatch for {dest_path.name}")
                    dest_path.unlink()
                    return False
            else:
                print(f"✓ {dest_path.name} downloaded")
                return True
                
        except Exception as e:
            print(f"✗ Error downloading {dest_path.name}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
            
    def verify_md5(self, file_path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of file"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest() == expected_md5
        
    def extract_gz(self, gz_path: Path, dest_path: Path):
        """Extract gzipped file"""
        print(f"Extracting {gz_path.name}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted to {dest_path.name}")
        
    def download_category(self, category: str, items: List[str] = None):
        """Download all items in a category"""
        if category not in DATA_SOURCES:
            print(f"Unknown category: {category}")
            return
            
        category_data = DATA_SOURCES[category]
        category_dir = self.base_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Filter items if specified
        if items:
            category_data = {k: v for k, v in category_data.items() if k in items}
            
        print(f"\n=== Downloading {category} ===")
        
        for name, info in category_data.items():
            if info.get('optional', False):
                print(f"\n{name}: {info['description']} (optional)")
                response = input("Download? (y/N): ")
                if response.lower() != 'y':
                    continue
                    
            print(f"\n{name}: {info['description']}")
            print(f"Size: {info.get('size', 'Unknown')}")
            
            # Special handling for different sources
            if info.get('format') == 'custom':
                print("⚠️  Requires manual download from:", info['url'])
                continue
                
            # Determine file extension
            url = info['url']
            if url.endswith('.gz'):
                dest_file = category_dir / f"{name}.gz"
                final_file = category_dir / name.replace('_', '.')
            else:
                ext = Path(url).suffix
                dest_file = category_dir / f"{name}{ext}"
                final_file = dest_file
                
            # Download
            success = self.download_file(
                url,
                dest_file,
                info.get('size'),
                info.get('md5')
            )
            
            # Extract if needed
            if success and dest_file.suffix == '.gz' and dest_file != final_file:
                self.extract_gz(dest_file, final_file)
                # Keep compressed version to save space
                # dest_file.unlink()

def download_from_ucsc(genome: str, data_type: str, output_dir: Path):
    """Download data from UCSC Genome Browser"""
    base_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}"
    
    files = {
        "chromsizes": f"{base_url}/bigZips/{genome}.chrom.sizes",
        "genes": f"{base_url}/bigZips/genes/{genome}.refGene.gtf.gz",
        "repeats": f"{base_url}/bigZips/{genome}.fa.out.gz",
        "mappability": f"{base_url}/encodeDCC/wgEncodeMapability/"
    }
    
    if data_type in files:
        url = files[data_type]
        filename = Path(url).name
        output_path = output_dir / filename
        
        downloader = DataDownloader()
        return downloader.download_file(url, output_path)
    else:
        print(f"Unknown data type: {data_type}")
        return False

def setup_blast_db(genome_path: Path):
    """Create BLAST database for sequence searches"""
    print("\nSetting up BLAST database...")
    
    try:
        # Check if makeblastdb is available
        result = subprocess.run(['makeblastdb', '-help'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Create BLAST database
            cmd = [
                'makeblastdb',
                '-in', str(genome_path),
                '-dbtype', 'nucl',
                '-parse_seqids',
                '-out', str(genome_path.with_suffix(''))
            ]
            
            subprocess.run(cmd, check=True)
            print("✓ BLAST database created")
        else:
            print("⚠️  makeblastdb not found. Install BLAST+ to enable sequence search.")
            
    except Exception as e:
        print(f"⚠️  Could not create BLAST database: {e}")

def download_models_from_huggingface():
    """Download model weights from HuggingFace"""
    print("\n=== Downloading Model Weights ===")
    
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Use huggingface-cli if available
    try:
        import huggingface_hub
        
        models = [
            ("LongSafari/hyenadna-medium-450k-seqlen", "hyenadna"),
            ("zhihan1996/DNABERT-2-117M", "dnabert2"),
            ("multimolecule/lucaone", "lucaone")
        ]
        
        for model_id, local_name in models:
            print(f"\nDownloading {model_id}...")
            local_dir = models_dir / local_name
            
            try:
                huggingface_hub.snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✓ Downloaded {local_name}")
            except Exception as e:
                print(f"✗ Error downloading {model_id}: {e}")
                
    except ImportError:
        print("⚠️  huggingface-hub not installed.")
        print("Install with: pip install huggingface-hub")
        print("Or download models manually from HuggingFace")

def main():
    parser = argparse.ArgumentParser(
        description="Download data and models for BRIDGE",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--category',
        choices=['all', 'genomes', 'annotations', 'regulatory', 
                'conservation', 'models'],
        default='all',
        help='Category of data to download'
    )
    
    parser.add_argument(
        '--items',
        nargs='+',
        help='Specific items to download (e.g., hg38 mm10)'
    )
    
    parser.add_argument(
        '--genomes',
        nargs='+',
        default=['hg38'],
        help='Genomes to download (default: hg38)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Base directory for data (default: data)'
    )
    
    parser.add_argument(
        '--models-only',
        action='store_true',
        help='Download only model weights'
    )
    
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Download minimal dataset for testing'
    )
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.models_only:
        download_models_from_huggingface()
        return
        
    if args.minimal:
        # Download minimal test dataset
        print("Downloading minimal test dataset...")
        categories = ['annotations']
        items = ['hg38_genes']
    elif args.category == 'all':
        categories = ['genomes', 'annotations', 'regulatory', 'models']
        items = None
    else:
        categories = [args.category]
        items = args.items
        
    # Download requested data
    for category in categories:
        if category == 'genomes' and args.genomes:
            # Filter genomes
            downloader.download_category(category, args.genomes)
        else:
            downloader.download_category(category, items)
            
    # Additional setup
    if 'genomes' in categories:
        for genome in args.genomes:
            genome_path = Path(args.data_dir) / 'genomes' / f"{genome}.fa"
            if genome_path.exists():
                setup_blast_db(genome_path)
                
    print("\n✅ Download complete!")
    print(f"Data saved to: {args.data_dir}/")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Download model weights: python download_data.py --models-only")
    print("2. Verify data integrity: python scripts/verify_data.py")
    print("3. Run tests: pytest tests/")
    print("4. Start analysis: bridge analyze -s <sequence>")

if __name__ == "__main__":
    main()
