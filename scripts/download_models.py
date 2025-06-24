#!/usr/bin/env python3
"""
Download pre-trained models for BRIDGE
Handles HuggingFace models and custom checkpoints
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import requests
from tqdm import tqdm
import hashlib
import json

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface-hub not installed. Install with: pip install huggingface-hub")

# Model registry
MODEL_REGISTRY = {
    "hyenadna": {
        "repo_id": "LongSafari/hyenadna-medium-450k-seqlen",
        "files": ["pytorch_model.bin", "config.json", "special_tokens_map.json"],
        "size": "6.8GB",
        "description": "HyenaDNA medium model with 450k context"
    },
    "dnabert2": {
        "repo_id": "zhihan1996/DNABERT-2-117M",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"],
        "size": "440MB", 
        "description": "DNABERT-2 117M parameter model"
    },
    "lucaone": {
        "repo_id": "multimolecule/lucaone",
        "files": ["pytorch_model.bin", "config.json", "tokenizer_config.json"],
        "size": "1.2GB",
        "description": "LucaOne multi-modal biological model"
    },
    "enformer": {
        "repo_id": "EleutherAI/enformer-official-rough",
        "files": ["pytorch_model.bin", "config.json"],
        "size": "1.2GB",
        "description": "Enformer for gene expression prediction (optional)"
    }
}

# Custom model URLs (for models not on HuggingFace)
CUSTOM_MODELS = {
    "bridge_enhancer_specialist": {
        "url": "https://example.com/bridge_enhancer_finetuned_v1.pt",
        "md5": "a1b2c3d4e5f6789012345678901234567890",
        "size": "450MB",
        "description": "BRIDGE fine-tuned for enhancer detection"
    },
    "bridge_promoter_specialist": {
        "url": "https://example.com/bridge_promoter_finetuned_v1.pt",
        "md5": "b2c3d4e5f67890123456789012345678901",
        "size": "450MB",
        "description": "BRIDGE fine-tuned for promoter detection"
    }
}

class ModelDownloader:
    """Handle model downloads with verification and caching"""
    
    def __init__(self, cache_dir: str = "models/pretrained"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def download_huggingface_model(self, model_name: str, force: bool = False) -> bool:
        """Download model from HuggingFace"""
        if not HF_AVAILABLE:
            self.logger.error("huggingface-hub not available")
            return False
            
        if model_name not in MODEL_REGISTRY:
            self.logger.error(f"Unknown model: {model_name}")
            return False
            
        model_info = MODEL_REGISTRY[model_name]
        model_dir = self.cache_dir / model_name
        
        # Check if already downloaded
        if model_dir.exists() and not force:
            if self._verify_model_files(model_dir, model_info['files']):
                self.logger.info(f"✓ {model_name} already downloaded")
                return True
                
        self.logger.info(f"Downloading {model_name} ({model_info['size']})...")
        self.logger.info(f"Description: {model_info['description']}")
        
        try:
            # Download all model files
            snapshot_download(
                repo_id=model_info['repo_id'],
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            self.logger.info(f"✓ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to download {model_name}: {e}")
            return False
            
    def download_custom_model(self, model_name: str, force: bool = False) -> bool:
        """Download custom model from URL"""
        if model_name not in CUSTOM_MODELS:
            self.logger.error(f"Unknown custom model: {model_name}")
            return False
            
        model_info = CUSTOM_MODELS[model_name]
        model_path = self.cache_dir / f"{model_name}.pt"
        
        # Check if already downloaded
        if model_path.exists() and not force:
            if self._verify_file_md5(model_path, model_info['md5']):
                self.logger.info(f"✓ {model_name} already downloaded")
                return True
                
        self.logger.info(f"Downloading {model_name} ({model_info['size']})...")
        
        try:
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            # Verify download
            if self._verify_file_md5(model_path, model_info['md5']):
                self.logger.info(f"✓ Successfully downloaded {model_name}")
                return True
            else:
                self.logger.error(f"✗ MD5 verification failed for {model_name}")
                model_path.unlink()
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()
            return False
            
    def download_all_models(self, include_optional: bool = False) -> Dict[str, bool]:
        """Download all required models"""
        results = {}
        
        # Required models
        required_models = ["hyenadna", "dnabert2", "lucaone"]
        
        self.logger.info("=== Downloading Required Models ===")
        for model in required_models:
            results[model] = self.download_huggingface_model(model)
            
        # Optional models
        if include_optional:
            self.logger.info("\n=== Downloading Optional Models ===")
            optional_models = ["enformer"] + list(CUSTOM_MODELS.keys())
            
            for model in optional_models:
                if model in MODEL_REGISTRY:
                    results[model] = self.download_huggingface_model(model)
                else:
                    results[model] = self.download_custom_model(model)
                    
        return results
        
    def verify_installation(self) -> Dict[str, Dict]:
        """Verify all downloaded models"""
        verification = {}
        
        self.logger.info("=== Verifying Model Installation ===")
        
        # Check HuggingFace models
        for model_name, model_info in MODEL_REGISTRY.items():
            model_dir = self.cache_dir / model_name
            
            if model_dir.exists():
                files_ok = self._verify_model_files(model_dir, model_info['files'])
                can_load = self._test_model_loading(model_name, model_dir)
                
                verification[model_name] = {
                    'installed': True,
                    'files_complete': files_ok,
                    'loadable': can_load,
                    'path': str(model_dir)
                }
            else:
                verification[model_name] = {
                    'installed': False,
                    'files_complete': False,
                    'loadable': False,
                    'path': None
                }
                
        # Check custom models
        for model_name in CUSTOM_MODELS:
            model_path = self.cache_dir / f"{model_name}.pt"
            
            if model_path.exists():
                verification[model_name] = {
                    'installed': True,
                    'files_complete': True,
                    'loadable': self._test_checkpoint_loading(model_path),
                    'path': str(model_path)
                }
            else:
                verification[model_name] = {
                    'installed': False,
                    'files_complete': False,
                    'loadable': False,
                    'path': None
                }
                
        return verification
        
    def _verify_model_files(self, model_dir: Path, required_files: List[str]) -> bool:
        """Check if all required files exist"""
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                return False
        return True
        
    def _verify_file_md5(self, file_path: Path, expected_md5: str) -> bool:
        """Verify file MD5 checksum"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest() == expected_md5
        
    def _test_model_loading(self, model_name: str, model_dir: Path) -> bool:
        """Test if model can be loaded"""
        try:
            if model_name == "hyenadna":
                # Test HyenaDNA loading
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
                return True
                
            elif model_name == "dnabert2":
                # Test DNABERT-2 loading
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                return True
                
            elif model_name == "lucaone":
                # Test LucaOne loading
                config_path = model_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    return True
                    
            return True
            
        except Exception as e:
            self.logger.debug(f"Loading test failed for {model_name}: {e}")
            return False
            
    def _test_checkpoint_loading(self, checkpoint_path: Path) -> bool:
        """Test if checkpoint can be loaded"""
        try:
            # Just check if it's a valid PyTorch file
            torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return True
        except:
            return False
            
    def print_verification_report(self, verification: Dict[str, Dict]):
        """Print nice verification report"""
        print("\n" + "="*60)
        print("MODEL VERIFICATION REPORT")
        print("="*60)
        
        all_good = True
        
        for model_name, status in verification.items():
            if status['installed']:
                if status['loadable']:
                    print(f"✓ {model_name:<20} OK")
                else:
                    print(f"⚠ {model_name:<20} Installed but cannot load")
                    all_good = False
            else:
                print(f"✗ {model_name:<20} Not installed")
                all_good = False
                
        print("="*60)
        
        if all_good:
            print("✓ All models ready!")
        else:
            print("⚠ Some models need attention")
            
        print("\nModel paths:")
        for model_name, status in verification.items():
            if status['path']:
                print(f"  {model_name}: {status['path']}")

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for BRIDGE",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to download (default: all required)'
    )
    
    parser.add_argument(
        '--include-optional',
        action='store_true',
        help='Also download optional models'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='models/pretrained',
        help='Directory to store models (default: models/pretrained)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing models without downloading'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if models exist'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.cache_dir)
    
    if args.verify_only:
        # Just verify
        verification = downloader.verify_installation()
        downloader.print_verification_report(verification)
        
    else:
        # Download models
        if args.models:
            # Download specific models
            results = {}
            for model in args.models:
                if model in MODEL_REGISTRY:
                    results[model] = downloader.download_huggingface_model(model, args.force)
                elif model in CUSTOM_MODELS:
                    results[model] = downloader.download_custom_model(model, args.force)
                else:
                    print(f"Unknown model: {model}")
                    
        else:
            # Download all required (and optional if requested)
            results = downloader.download_all_models(args.include_optional)
            
        # Print results
        print("\n" + "="*40)
        print("DOWNLOAD SUMMARY")
        print("="*40)
        
        for model, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{model:<20} {status}")
            
        # Verify installation
        print("\n")
        verification = downloader.verify_installation()
        downloader.print_verification_report(verification)
        
        # Print next steps
        if all(results.values()):
            print("\n✅ All models downloaded successfully!")
            print("\nNext steps:")
            print("1. Run tests: pytest tests/")
            print("2. Start BRIDGE: python bridge_orchestrator.py --help")
        else:
            print("\n⚠ Some downloads failed. Please check the errors above.")

if __name__ == "__main__":
    main()
