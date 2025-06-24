#!/usr/bin/env python3
"""
Fine-tune BRIDGE models on known enhancer/promoter sequences
Supports multiple data sources and training strategies
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import track
import pybedtools
import pyfaidx

console = Console()

# ======================== Data Loaders ========================

class RegulatoryDatasetBuilder:
    """Build training datasets from various sources"""
    
    def __init__(self, genome_fasta: str):
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.console = console
        
    def load_from_encode(self, encode_peaks: str, element_type: str) -> Dict:
        """
        Load from ENCODE ChIP-seq peaks
        Common files:
        - H3K27ac for active enhancers
        - H3K4me3 for active promoters
        - P300/CBP binding for enhancers
        """
        self.console.print(f"Loading {element_type} regions from ENCODE peaks...")
        
        peaks = pd.read_csv(encode_peaks, sep='\t', header=None,
                           names=['chr', 'start', 'end', 'name', 'score'])
        
        # Filter high-confidence peaks
        high_conf = peaks[peaks['score'] > peaks['score'].quantile(0.75)]
        
        sequences = []
        labels = []
        
        for _, peak in track(high_conf.iterrows(), description="Extracting sequences"):
            # Extend peak to typical element size
            if element_type == 'enhancer':
                center = (peak['start'] + peak['end']) // 2
                start = center - 500
                end = center + 500
            else:  # promoter
                # Promoters are usually smaller
                center = (peak['start'] + peak['end']) // 2
                start = center - 250
                end = center + 250
                
            # Extract sequence
            try:
                seq = self.genome[peak['chr']][start:end].seq.upper()
                
                sequences.append({
                    'sequence': seq,
                    'chromosome': peak['chr'],
                    'start': start,
                    'end': end
                })
                
                labels.append({
                    'elements': [{
                        'type': element_type,
                        'start': 0,
                        'end': len(seq),
                        'score': float(peak['score']) / 1000  # Normalize
                    }]
                })
            except:
                continue
                
        self.console.print(f"Loaded {len(sequences)} {element_type} sequences")
        return {'sequences': sequences, 'labels': labels}
        
    def load_from_vista(self, vista_file: str) -> Dict:
        """
        Load from VISTA Enhancer Browser
        Experimentally validated enhancers
        """
        self.console.print("Loading VISTA validated enhancers...")
        
        vista_data = pd.read_csv(vista_file, sep='\t')
        
        sequences = []
        labels = []
        
        for _, enhancer in vista_data.iterrows():
            if enhancer['validation'] == 'positive':
                try:
                    seq = self.genome[enhancer['chr']][
                        enhancer['start']:enhancer['end']
                    ].seq.upper()
                    
                    sequences.append({
                        'sequence': seq,
                        'chromosome': enhancer['chr'],
                        'start': enhancer['start'],
                        'end': enhancer['end'],
                        'tissues': enhancer['tissues']
                    })
                    
                    labels.append({
                        'elements': [{
                            'type': 'enhancer',
                            'start': 0,
                            'end': len(seq),
                            'validated': True,
                            'active_tissues': enhancer['tissues'].split(',')
                        }]
                    })
                except:
                    continue
                    
        self.console.print(f"Loaded {len(sequences)} validated enhancers")
        return {'sequences': sequences, 'labels': labels}
        
    def load_from_fantom5(self, fantom_file: str) -> Dict:
        """
        Load from FANTOM5 CAGE peaks
        Precise TSS and enhancer locations
        """
        self.console.print("Loading FANTOM5 regulatory elements...")
        
        # FANTOM5 provides precise TSS locations
        cage_peaks = pd.read_csv(fantom_file, sep='\t')
        
        sequences = []
        labels = []
        
        for _, peak in cage_peaks.iterrows():
            # Determine if promoter or enhancer based on distance to genes
            element_type = 'promoter' if peak['distance_to_tss'] < 500 else 'enhancer'
            
            try:
                seq = self.genome[peak['chr']][
                    peak['start']:peak['end']
                ].seq.upper()
                
                sequences.append({
                    'sequence': seq,
                    'chromosome': peak['chr'],
                    'start': peak['start'],
                    'end': peak['end']
                })
                
                labels.append({
                    'elements': [{
                        'type': element_type,
                        'start': 0,
                        'end': len(seq),
                        'expression_correlation': peak['expression_correlation']
                    }]
                })
            except:
                continue
                
        return {'sequences': sequences, 'labels': labels}
        
    def generate_negatives(self, positive_data: Dict, strategy: str = 'mixed') -> Dict:
        """
        Generate negative examples using various strategies
        """
        self.console.print(f"Generating negative examples using {strategy} strategy...")
        
        num_positives = len(positive_data['sequences'])
        negative_sequences = []
        negative_labels = []
        
        if strategy == 'shuffle' or strategy == 'mixed':
            # Shuffle positive sequences (preserves composition)
            for i in track(range(num_positives // 3), description="Shuffling sequences"):
                pos_seq = positive_data['sequences'][i]['sequence']
                shuffled = ''.join(np.random.permutation(list(pos_seq)))
                
                negative_sequences.append({
                    'sequence': shuffled,
                    'chromosome': 'shuffled',
                    'start': 0,
                    'end': len(shuffled)
                })
                
                negative_labels.append({'elements': []})
                
        if strategy == 'random' or strategy == 'mixed':
            # Random genomic regions
            chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
            
            for i in track(range(num_positives // 3), description="Random regions"):
                chr_name = np.random.choice(chromosomes)
                chr_len = len(self.genome[chr_name])
                
                # Random position
                start = np.random.randint(0, chr_len - 1000)
                end = start + 1000
                
                try:
                    seq = self.genome[chr_name][start:end].seq.upper()
                    
                    # Skip if too many Ns
                    if seq.count('N') / len(seq) > 0.1:
                        continue
                        
                    negative_sequences.append({
                        'sequence': seq,
                        'chromosome': chr_name,
                        'start': start,
                        'end': end
                    })
                    
                    negative_labels.append({'elements': []})
                except:
                    continue
                    
        if strategy == 'dinucleotide' or strategy == 'mixed':
            # Dinucleotide-preserving shuffle
            for i in track(range(num_positives // 3), description="Dinucleotide shuffle"):
                pos_seq = positive_data['sequences'][i]['sequence']
                shuffled = self._dinucleotide_shuffle(pos_seq)
                
                negative_sequences.append({
                    'sequence': shuffled,
                    'chromosome': 'di_shuffled',
                    'start': 0,
                    'end': len(shuffled)
                })
                
                negative_labels.append({'elements': []})
                
        return {
            'sequences': negative_sequences,
            'labels': negative_labels
        }
        
    def _dinucleotide_shuffle(self, sequence: str) -> str:
        """Shuffle preserving dinucleotide frequencies"""
        # Build dinucleotide graph
        dinucs = [sequence[i:i+2] for i in range(len(sequence)-1)]
        np.random.shuffle(dinucs)
        
        # Reconstruct sequence
        result = dinucs[0][0]
        for di in dinucs:
            result += di[1]
            
        return result

# ======================== Training Configuration ========================

class TrainingConfig:
    """Configuration for fine-tuning"""
    
    def __init__(self):
        # Model settings
        self.model_configs = {
            'dnabert2': {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'max_length': 512,
                'lora_r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            },
            'hyenadna': {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'max_length': 8192,
                'lora_r': 32,
                'lora_alpha': 64,
                'lora_dropout': 0.1
            }
        }
        
        # Training settings
        self.training = {
            'epochs': 10,
            'warmup_steps': 0.1,  # Fraction of total steps
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'eval_steps': 100,
            'save_steps': 500,
            'logging_steps': 10
        }
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'classification': 1.0,
            'boundary': 0.5,
            'confidence': 0.3
        }

# ======================== Custom Loss Functions ========================

class RegulatoryLoss(torch.nn.Module):
    """Custom loss for regulatory element detection"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Base losses
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse = torch.nn.MSELoss()
        
        # Class weights (enhancers/promoters are rare)
        self.pos_weight = torch.tensor([10.0])  # Weight for positive class
        
    def forward(self, predictions, targets, attention_weights=None):
        """
        Compute multi-component loss
        
        Args:
            predictions: Dict with 'logits', 'boundaries', 'confidence'
            targets: Dict with 'labels', 'boundaries', 'masks'
        """
        total_loss = 0.0
        losses = {}
        
        # 1. Classification loss (element vs background)
        class_loss = self.bce(
            predictions['logits'],
            targets['labels']
        )
        
        # Apply position weighting (weight boundaries higher)
        if 'boundary_mask' in targets:
            boundary_weight = 1.0 + 2.0 * targets['boundary_mask']
            class_loss = class_loss * boundary_weight
            
        # Apply class weighting
        pos_mask = targets['labels'] > 0.5
        class_loss[pos_mask] *= self.pos_weight
        
        losses['classification'] = class_loss.mean()
        total_loss += self.config.loss_weights['classification'] * losses['classification']
        
        # 2. Boundary refinement loss
        if 'boundaries' in predictions and 'boundaries' in targets:
            boundary_loss = self.mse(
                predictions['boundaries'],
                targets['boundaries']
            )
            losses['boundary'] = boundary_loss
            total_loss += self.config.loss_weights['boundary'] * boundary_loss
            
        # 3. Confidence calibration loss
        if 'confidence' in predictions:
            # Confidence should be high for correct predictions
            correct = (predictions['logits'] > 0).float() == targets['labels']
            target_confidence = correct.float()
            
            confidence_loss = self.mse(
                predictions['confidence'],
                target_confidence
            )
            losses['confidence'] = confidence_loss
            total_loss += self.config.loss_weights['confidence'] * confidence_loss
            
        # 4. Attention consistency loss (optional)
        if attention_weights is not None:
            # Attention should focus on element regions
            attention_loss = self._attention_consistency_loss(
                attention_weights,
                targets['labels']
            )
            losses['attention'] = attention_loss
            total_loss += 0.1 * attention_loss
            
        return total_loss, losses
        
    def _attention_consistency_loss(self, attention, labels):
        """Attention should focus on positive regions"""
        # Average attention across heads and layers
        avg_attention = attention.mean(dim=(1, 2))  # (batch, seq_len)
        
        # Attention should be high where labels are positive
        target_attention = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
        
        return self.mse(avg_attention, target_attention)

# ======================== Fine-tuning Pipeline ========================

def finetune_model(args):
    """Main fine-tuning pipeline"""
    console.print("[bold blue]BRIDGE Model Fine-tuning Pipeline[/bold blue]")
    
    # Load configuration
    config = TrainingConfig()
    
    # Build dataset
    builder = RegulatoryDatasetBuilder(args.genome_fasta)
    
    # Load positive examples
    all_sequences = []
    all_labels = []
    
    if args.encode_peaks:
        data = builder.load_from_encode(args.encode_peaks, args.element_type)
        all_sequences.extend(data['sequences'])
        all_labels.extend(data['labels'])
        
    if args.vista_file:
        data = builder.load_from_vista(args.vista_file)
        all_sequences.extend(data['sequences'])
        all_labels.extend(data['labels'])
        
    if args.fantom_file:
        data = builder.load_from_fantom5(args.fantom_file)
        all_sequences.extend(data['sequences'])
        all_labels.extend(data['labels'])
        
    console.print(f"Loaded {len(all_sequences)} positive examples")
    
    # Generate negatives
    positive_data = {'sequences': all_sequences, 'labels': all_labels}
    negative_data = builder.generate_negatives(
        positive_data,
        strategy=args.negative_strategy
    )
    
    # Combine positive and negative
    all_sequences.extend(negative_data['sequences'])
    all_labels.extend(negative_data['labels'])
    
    console.print(f"Total dataset size: {len(all_sequences)}")
    
    # Split train/val/test
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        all_sequences, all_labels, test_size=0.2, random_state=42, stratify=[bool(l['elements']) for l in all_labels]
    )
    
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.1, random_state=42
    )
    
    console.print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'training_data.json', 'w') as f:
        json.dump({
            'train': {'sequences': train_sequences, 'labels': train_labels},
            'val': {'sequences': val_sequences, 'labels': val_labels},
            'test': {'sequences': test_sequences, 'labels': test_labels},
            'metadata': {
                'element_type': args.element_type,
                'positive_sources': {
                    'encode': args.encode_peaks,
                    'vista': args.vista_file,
                    'fantom': args.fantom_file
                },
                'negative_strategy': args.negative_strategy
            }
        }, f, indent=2)
        
    # Initialize model and trainer
    from bridge_train import BridgeTrainer
    from bridge_models import DNABert2Wrapper, HyenaDNAWrapper
    
    if args.model == 'dnabert2':
        model_wrapper = DNABert2Wrapper()
    elif args.model == 'hyenadna':
        model_wrapper = HyenaDNAWrapper()
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Create trainer
    trainer = BridgeTrainer(
        model_wrapper,
        args.model,
        task='regulatory',
        output_dir=str(output_dir),
        use_lora=not args.no_lora
    )
    
    # Prepare datasets
    from bridge_train import RegulatoryElementDataset
    
    train_dataset = RegulatoryElementDataset(
        sequences=[s['sequence'] for s in train_sequences],
        labels=train_labels,
        tokenizer=model_wrapper.tokenizer,
        max_length=config.model_configs[args.model]['max_length']
    )
    
    val_dataset = RegulatoryElementDataset(
        sequences=[s['sequence'] for s in val_sequences],
        labels=val_labels,
        tokenizer=model_wrapper.tokenizer,
        max_length=config.model_configs[args.model]['max_length']
    )
    
    # Fine-tune
    console.print("\n[bold]Starting fine-tuning...[/bold]")
    
    trainer.train_regulatory_detection(
        train_dataset,
        val_dataset,
        epochs=args.epochs or config.training['epochs'],
        learning_rate=config.model_configs[args.model]['learning_rate'],
        batch_size=config.model_configs[args.model]['batch_size']
    )
    
    # Evaluate on test set
    console.print("\n[bold]Evaluating on test set...[/bold]")
    
    test_dataset = RegulatoryElementDataset(
        sequences=[s['sequence'] for s in test_sequences],
        labels=test_labels,
        tokenizer=model_wrapper.tokenizer,
        max_length=config.model_configs[args.model]['max_length']
    )
    
    # Run evaluation
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_wrapper.model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(model_wrapper.device) for k, v in batch.items() if k != 'labels'}
            outputs = model_wrapper.model(**inputs)
            
            # Get predictions
            predictions = torch.sigmoid(outputs.logits)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
    # Calculate metrics
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    
    # Calculate final metrics
    predictions_binary = all_predictions > best_threshold
    tp = np.sum((predictions_binary == 1) & (all_labels == 1))
    fp = np.sum((predictions_binary == 1) & (all_labels == 0))
    fn = np.sum((predictions_binary == 0) & (all_labels == 1))
    
    final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    
    console.print("\n[bold green]Test Set Results:[/bold green]")
    console.print(f"Precision: {final_precision:.3f}")
    console.print(f"Recall: {final_recall:.3f}")
    console.print(f"F1 Score: {final_f1:.3f}")
    console.print(f"Optimal threshold: {best_threshold:.3f}")
    console.print(f"Average Precision: {average_precision_score(all_labels, all_predictions):.3f}")
    
    # Save evaluation results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'test_metrics': {
                'precision': float(final_precision),
                'recall': float(final_recall),
                'f1_score': float(final_f1),
                'average_precision': float(average_precision_score(all_labels, all_predictions)),
                'optimal_threshold': float(best_threshold)
            },
            'model': args.model,
            'element_type': args.element_type
        }, f, indent=2)
        
    console.print(f"\n[green]✓ Fine-tuning complete! Model saved to {output_dir}[/green]")

# ======================== Main Entry Point ========================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BRIDGE models on known regulatory elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune on ENCODE H3K27ac peaks (enhancers)
  python bridge_finetune.py \\
    --encode-peaks h3k27ac_peaks.bed \\
    --element-type enhancer \\
    --genome-fasta hg38.fa \\
    --model dnabert2
    
  # Fine-tune on VISTA validated enhancers
  python bridge_finetune.py \\
    --vista-file vista_enhancers.txt \\
    --genome-fasta mm10.fa \\
    --model hyenadna
    
  # Combined training with multiple sources
  python bridge_finetune.py \\
    --encode-peaks h3k4me3_peaks.bed \\
    --fantom-file fantom5_cage.txt \\
    --element-type promoter \\
    --genome-fasta hg38.fa
        """
    )
    
    # Data sources
    parser.add_argument('--encode-peaks', help='ENCODE ChIP-seq peaks (BED format)')
    parser.add_argument('--vista-file', help='VISTA enhancer browser file')
    parser.add_argument('--fantom-file', help='FANTOM5 CAGE peaks file')
    parser.add_argument('--custom-bed', help='Custom BED file with validated elements')
    
    # Required arguments
    parser.add_argument('--genome-fasta', required=True, help='Reference genome FASTA')
    parser.add_argument('--element-type', required=True, 
                       choices=['enhancer', 'promoter', 'both'],
                       help='Type of regulatory element')
    
    # Model settings
    parser.add_argument('--model', default='dnabert2',
                       choices=['dnabert2', 'hyenadna'],
                       help='Model to fine-tune')
    parser.add_argument('--no-lora', action='store_true',
                       help='Disable LoRA (full fine-tuning)')
    
    # Training settings
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--negative-strategy', default='mixed',
                       choices=['shuffle', 'random', 'dinucleotide', 'mixed'],
                       help='Strategy for generating negative examples')
    
    # Output
    parser.add_argument('--output-dir', default='models/finetuned',
                       help='Output directory for model and results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.encode_peaks, args.vista_file, args.fantom_file, args.custom_bed]):
        parser.error("At least one data source must be provided")
        
    # Run fine-tuning
    finetune_model(args)

if __name__ == "__main__":
    main()
