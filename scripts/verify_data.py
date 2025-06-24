#!/usr/bin/env python3
"""
Verify BRIDGE data integrity and completeness
Checks genomes, annotations, and other required data files
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from datetime import datetime

try:
    import pyfaidx
    FAIDX_AVAILABLE = True
except ImportError:
    FAIDX_AVAILABLE = False
    print("Warning: pyfaidx not installed. Install with: pip install pyfaidx")

try:
    import pyBigWig
    BIGWIG_AVAILABLE = True
except ImportError:
    BIGWIG_AVAILABLE = False
    print("Warning: pyBigWig not installed. Install with: pip install pyBigWig")

# Expected data files and their properties
EXPECTED_DATA = {
    "genomes": {
        "hg38.fa": {
            "size_min": 3000000000,  # ~3GB
            "size_max": 3300000000,
            "chromosomes": ["chr1", "chr2", "chr3", "chr22", "chrX", "chrY"],
            "description": "Human reference genome (GRCh38)",
            "required": True
        },
        "mm10.fa": {
            "size_min": 2600000000,  # ~2.6GB
            "size_max": 2900000000,
            "chromosomes": ["chr1", "chr2", "chr19", "chrX", "chrY"],
            "description": "Mouse reference genome",
            "required": False
        },
        "danRer11.fa": {
            "size_min": 1300000000,  # ~1.3GB
            "size_max": 1500000000,
            "chromosomes": ["chr1", "chr2", "chr25"],
            "description": "Zebrafish reference genome",
            "required": False
        }
    },
    "annotations": {
        "hg38.refGene.gtf": {
            "size_min": 50000000,   # ~50MB
            "size_max": 200000000,  # ~200MB
            "description": "Human gene annotations",
            "required": True,
            "format": "gtf"
        },
        "cpgIslandExt.txt": {
            "size_min": 2000000,    # ~2MB
            "size_max": 5000000,    # ~5MB
            "description": "CpG island annotations",
            "required": False,
            "format": "bed"
        }
    },
    "regulatory": {
        "encode_cCREs.bed": {
            "size_min": 100000000,  # ~100MB
            "size_max": 500000000,  # ~500MB
            "description": "ENCODE candidate regulatory elements",
            "required": False,
            "format": "bed"
        },
        "fantom5_cage.bed": {
            "size_min": 30000000,   # ~30MB
            "size_max": 100000000,  # ~100MB
            "description": "FANTOM5 CAGE peaks",
            "required": False,
            "format": "bed"
        }
    },
    "conservation": {
        "hg38.phyloP100way.bw": {
            "size_min": 9000000000,  # ~9GB
            "size_max": 11000000000, # ~11GB
            "description": "100-way vertebrate conservation (phyloP)",
            "required": False,
            "format": "bigwig"
        },
        "hg38.phastCons100way.bw": {
            "size_min": 1000000000,  # ~1GB
            "size_max": 1500000000,  # ~1.5GB
            "description": "100-way vertebrate conservation (phastCons)",
            "required": False,
            "format": "bigwig"
        }
    }
}

class DataVerifier:
    """Verify data integrity and completeness"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        self.verification_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def verify_all(self) -> Dict:
        """Run all verification checks"""
        self.logger.info("Starting BRIDGE data verification...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'categories': {}
        }
        
        # Check each category
        for category, files in EXPECTED_DATA.items():
            self.logger.info(f"\n=== Verifying {category} ===")
            category_results = self.verify_category(category, files)
            results['categories'][category] = category_results
            
        # Summary
        results['summary'] = self._generate_summary(results['categories'])
        
        return results
        
    def verify_category(self, category: str, expected_files: Dict) -> Dict:
        """Verify all files in a category"""
        category_dir = self.data_dir / category
        results = {
            'exists': category_dir.exists(),
            'files': {}
        }
        
        if not category_dir.exists():
            self.logger.warning(f"Category directory not found: {category_dir}")
            return results
            
        for filename, properties in expected_files.items():
            file_path = category_dir / filename
            file_results = self.verify_file(file_path, properties)
            results['files'][filename] = file_results
            
            # Log result
            if file_results['exists']:
                if file_results['valid']:
                    self.logger.info(f"✓ {filename}")
                else:
                    self.logger.warning(f"⚠ {filename} - {file_results.get('error', 'Invalid')}")
            else:
                if properties.get('required', False):
                    self.logger.error(f"✗ {filename} - Required file missing")
                else:
                    self.logger.info(f"- {filename} - Optional file not found")
                    
        return results
        
    def verify_file(self, file_path: Path, properties: Dict) -> Dict:
        """Verify individual file"""
        results = {
            'exists': file_path.exists(),
            'path': str(file_path),
            'required': properties.get('required', False),
            'description': properties.get('description', '')
        }
        
        if not file_path.exists():
            results['valid'] = False
            return results
            
        # Check file size
        file_size = file_path.stat().st_size
        results['size'] = file_size
        
        size_min = properties.get('size_min', 0)
        size_max = properties.get('size_max', float('inf'))
        
        if not (size_min <= file_size <= size_max):
            results['valid'] = False
            results['error'] = f"Size {file_size:,} bytes outside expected range"
            return results
            
        # Format-specific checks
        file_format = properties.get('format', '')
        
        try:
            if file_format == 'gtf':
                results.update(self._verify_gtf(file_path))
            elif file_format == 'bed':
                results.update(self._verify_bed(file_path))
            elif file_format == 'bigwig' and BIGWIG_AVAILABLE:
                results.update(self._verify_bigwig(file_path))
            elif file_path.suffix == '.fa' and FAIDX_AVAILABLE:
                results.update(self._verify_fasta(file_path, properties))
            else:
                results['valid'] = True
                
        except Exception as e:
            results['valid'] = False
            results['error'] = str(e)
            
        return results
        
    def _verify_gtf(self, file_path: Path) -> Dict:
        """Verify GTF file format"""
        results = {'valid': True}
        
        try:
            line_count = 0
            gene_count = 0
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                        
                    line_count += 1
                    parts = line.strip().split('\t')
                    
                    if len(parts) >= 9:
                        if parts[2] == 'gene':
                            gene_count += 1
                            
                    if line_count >= 1000:  # Sample first 1000 lines
                        break
                        
            results['line_count'] = line_count
            results['gene_count'] = gene_count
            
            if gene_count == 0:
                results['valid'] = False
                results['error'] = "No gene entries found"
                
        except Exception as e:
            results['valid'] = False
            results['error'] = f"GTF parsing error: {e}"
            
        return results
        
    def _verify_bed(self, file_path: Path) -> Dict:
        """Verify BED file format"""
        results = {'valid': True}
        
        try:
            line_count = 0
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.startswith('track'):
                        continue
                        
                    line_count += 1
                    parts = line.strip().split('\t')
                    
                    if len(parts) < 3:
                        results['valid'] = False
                        results['error'] = f"Invalid BED format at line {line_count}"
                        break
                        
                    # Check coordinates
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        if start >= end:
                            results['valid'] = False
                            results['error'] = f"Invalid coordinates at line {line_count}"
                            break
                    except ValueError:
                        results['valid'] = False
                        results['error'] = f"Non-numeric coordinates at line {line_count}"
                        break
                        
                    if line_count >= 1000:  # Sample first 1000 lines
                        break
                        
            results['line_count'] = line_count
            
        except Exception as e:
            results['valid'] = False
            results['error'] = f"BED parsing error: {e}"
            
        return results
        
    def _verify_bigwig(self, file_path: Path) -> Dict:
        """Verify BigWig file"""
        results = {'valid': True}
        
        try:
            bw = pyBigWig.open(str(file_path))
            
            # Check if file is valid
            if not bw.isBigWig():
                results['valid'] = False
                results['error'] = "Not a valid BigWig file"
            else:
                # Get some stats
                results['chromosomes'] = list(bw.chroms().keys())[:5]  # First 5
                results['total_coverage'] = sum(bw.chroms().values())
                
            bw.close()
            
        except Exception as e:
            results['valid'] = False
            results['error'] = f"BigWig error: {e}"
            
        return results
        
    def _verify_fasta(self, file_path: Path, properties: Dict) -> Dict:
        """Verify FASTA file"""
        results = {'valid': True}
        
        try:
            # Check if index exists
            fai_path = Path(str(file_path) + '.fai')
            results['indexed'] = fai_path.exists()
            
            # Open FASTA
            fasta = pyfaidx.Fasta(str(file_path))
            
            # Check chromosomes
            available_chroms = list(fasta.keys())
            results['chromosomes'] = available_chroms[:10]  # First 10
            results['total_chromosomes'] = len(available_chroms)
            
            # Check expected chromosomes if specified
            expected_chroms = properties.get('chromosomes', [])
            if expected_chroms:
                missing_chroms = set(expected_chroms) - set(available_chroms)
                if missing_chroms:
                    results['valid'] = False
                    results['error'] = f"Missing chromosomes: {missing_chroms}"
                    
            # Calculate total size
            total_size = sum(len(fasta[chrom]) for chrom in available_chroms)
            results['total_bases'] = total_size
            
            fasta.close()
            
        except Exception as e:
            results['valid'] = False
            results['error'] = f"FASTA error: {e}"
            
        return results
        
    def _generate_summary(self, categories: Dict) -> Dict:
        """Generate verification summary"""
        summary = {
            'total_files': 0,
            'files_found': 0,
            'files_valid': 0,
            'required_missing': [],
            'errors': []
        }
        
        for category, cat_data in categories.items():
            if not cat_data.get('exists', False):
                continue
                
            for filename, file_data in cat_data.get('files', {}).items():
                summary['total_files'] += 1
                
                if file_data['exists']:
                    summary['files_found'] += 1
                    
                    if file_data.get('valid', False):
                        summary['files_valid'] += 1
                    else:
                        summary['errors'].append({
                            'file': f"{category}/{filename}",
                            'error': file_data.get('error', 'Unknown error')
                        })
                else:
                    if file_data.get('required', False):
                        summary['required_missing'].append(f"{category}/{filename}")
                        
        summary['all_required_present'] = len(summary['required_missing']) == 0
        summary['all_valid'] = summary['files_valid'] == summary['files_found']
        
        return summary
        
    def print_report(self, results: Dict):
        """Print verification report"""
        print("\n" + "="*60)
        print("BRIDGE DATA VERIFICATION REPORT")
        print("="*60)
        print(f"Data directory: {results['data_dir']}")
        print(f"Timestamp: {results['timestamp']}")
        print("="*60)
        
        # Print category results
        for category, cat_data in results['categories'].items():
            print(f"\n{category.upper()}:")
            
            if not cat_data['exists']:
                print(f"  ✗ Directory not found")
                continue
                
            for filename, file_data in cat_data['files'].items():
                status = "✓" if file_data.get('valid', False) else "✗"
                required = " (required)" if file_data.get('required', False) else ""
                
                if file_data['exists']:
                    size_mb = file_data.get('size', 0) / (1024 * 1024)
                    print(f"  {status} {filename}{required} - {size_mb:.1f} MB")
                    
                    if not file_data.get('valid', False):
                        print(f"     Error: {file_data.get('error', 'Unknown')}")
                else:
                    print(f"  - {filename}{required} - Not found")
                    
        # Print summary
        summary = results['summary']
        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"  Total files checked: {summary['total_files']}")
        print(f"  Files found: {summary['files_found']}")
        print(f"  Files valid: {summary['files_valid']}")
        
        if summary['required_missing']:
            print(f"\n  ⚠ MISSING REQUIRED FILES:")
            for f in summary['required_missing']:
                print(f"    - {f}")
                
        if summary['errors']:
            print(f"\n  ⚠ VALIDATION ERRORS:")
            for err in summary['errors']:
                print(f"    - {err['file']}: {err['error']}")
                
        print("="*60)
        
        if summary['all_required_present'] and summary['all_valid']:
            print("✅ All data files verified successfully!")
        elif summary['all_required_present']:
            print("⚠️  All required files present but some have errors")
        else:
            print("❌ Missing required files - please run download scripts")
            
    def generate_md5_checksums(self, output_file: str = "data_checksums.json"):
        """Generate MD5 checksums for all data files"""
        checksums = {}
        
        self.logger.info("Generating MD5 checksums...")
        
        for category in EXPECTED_DATA:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
                
            checksums[category] = {}
            
            for file_path in category_dir.glob("*"):
                if file_path.is_file():
                    self.logger.info(f"  Calculating checksum for {file_path.name}...")
                    md5 = self._calculate_md5(file_path)
                    checksums[category][file_path.name] = {
                        'md5': md5,
                        'size': file_path.stat().st_size
                    }
                    
        # Save checksums
        with open(output_file, 'w') as f:
            json.dump(checksums, f, indent=2)
            
        self.logger.info(f"Checksums saved to {output_file}")
        
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        md5_hash = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                
        return md5_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser(
        description="Verify BRIDGE data files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory to verify (default: data)'
    )
    
    parser.add_argument(
        '--save-report',
        help='Save verification report to JSON file'
    )
    
    parser.add_argument(
        '--generate-checksums',
        action='store_true',
        help='Generate MD5 checksums for all files'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = DataVerifier(args.data_dir)
    
    # Run verification
    results = verifier.verify_all()
    
    # Print report
    if not args.quiet:
        verifier.print_report(results)
        
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to: {args.save_report}")
        
    # Generate checksums if requested
    if args.generate_checksums:
        verifier.generate_md5_checksums()
        
    # Exit code based on verification
    if results['summary']['all_required_present']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
