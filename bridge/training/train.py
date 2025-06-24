"""
BRIDGE Training Module - Fine-tune models for specific regulatory tasks
Supports LoRA and full fine-tuning with biological constraints
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType

from bridge_models import DNASequence, RegulatoryElement

# ======================== Datasets ========================

class RegulatoryElementDataset(Dataset):
    """Dataset for regulatory element detection"""
    
    def __init__(self, sequences: List[str], labels: List[Dict],
                 tokenizer, max_length: int = 512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create label tensor
        # Binary classification for each position
        label_tensor = torch.zeros(self.max_length)
        
        if 'elements' in label:
            for element in label['elements']:
                start = element['start']
                end = element['end']
                
                # Map to token positions
                # This is simplified - actual implementation would
                # properly map sequence positions to token positions
                label_tensor[start:end] = 1.0
                
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_tensor
        }

class ExpressionDataset(Dataset):
    """Dataset for expression prediction"""
    
    def __init__(self, gene_sequences: List[Dict], 
                 expression_values: List[float],
                 tokenizer, max_length: int = 8192):
        self.gene_sequences = gene_sequences
        self.expression_values = expression_values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.gene_sequences)
        
    def __getitem__(self, idx):
        gene_data = self.gene_sequences[idx]
        expression = self.expression_values[idx]
        
        # Include promoter + gene body
        sequence = gene_data['promoter'] + gene_data['gene_body'][:5000]
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'expression': torch.tensor(expression, dtype=torch.float)
        }

# ======================== Training Functions ========================

class BridgeTrainer:
    """Main trainer for BRIDGE models"""
    
    def __init__(self, model, model_name: str, task: str,
                 output_dir: str, use_lora: bool = True):
        self.model = model
        self.model_name = model_name
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_lora = use_lora
        
        # Setup LoRA if requested
        if use_lora:
            self._setup_lora()
            
    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=self._get_target_modules(),
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.TOKEN_CLS if self.task == 'regulatory' else TaskType.SEQ_2_SEQ_LM
        )
        
        self.model = get_peft_model(self.model.model, lora_config)
        self.model.print_trainable_parameters()
        
    def _get_target_modules(self):
        """Get model-specific target modules for LoRA"""
        if 'dnabert' in self.model_name.lower():
            return ["query", "value"]
        elif 'hyena' in self.model_name.lower():
            return ["in_proj", "out_proj"]
        elif 'lucaone' in self.model_name.lower():
            return ["q_proj", "v_proj"]
        else:
            return ["query", "value"]  # default
            
    def train_regulatory_detection(self, train_dataset: RegulatoryElementDataset,
                                 val_dataset: RegulatoryElementDataset,
                                 epochs: int = 10,
                                 learning_rate: float = 2e-5,
                                 batch_size: int = 16):
        """Train model for regulatory element detection"""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4
        )
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
        
        # Training loop
        best_val_f1 = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get logits from hidden states
                hidden_states = outputs.last_hidden_state
                
                # Simple linear projection for classification
                if not hasattr(self, 'classifier'):
                    self.classifier = nn.Linear(
                        hidden_states.shape[-1], 1
                    ).to(self.model.device)
                    
                logits = self.classifier(hidden_states).squeeze(-1)
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
            # Validation
            val_metrics = self._validate_regulatory(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self._save_checkpoint(epoch, val_metrics)
                
    def _validate_regulatory(self, val_loader):
        """Validate regulatory element detection"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                hidden_states = outputs.last_hidden_state
                logits = self.classifier(hidden_states).squeeze(-1)
                
                # Get predictions
                preds = torch.sigmoid(logits) > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def train_expression_prediction(self, train_dataset: ExpressionDataset,
                                  val_dataset: ExpressionDataset,
                                  epochs: int = 10,
                                  learning_rate: float = 1e-5,
                                  batch_size: int = 8):
        """Train model for expression prediction"""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4
        )
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Regression head
        if not hasattr(self, 'regressor'):
            hidden_size = self.model.config.hidden_size
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ).to(self.model.device)
            
        # Training loop
        best_val_corr = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                expression = batch['expression'].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Pool hidden states
                hidden_states = outputs.last_hidden_state
                pooled = hidden_states.mean(dim=1)
                
                # Predict expression
                pred_expression = self.regressor(pooled).squeeze()
                
                # Compute loss
                loss = criterion(pred_expression, expression)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation
            val_metrics = self._validate_expression(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Correlation: {val_metrics['correlation']:.4f}")
            print(f"Val MSE: {val_metrics['mse']:.4f}")
            
            # Save best model
            if val_metrics['correlation'] > best_val_corr:
                best_val_corr = val_metrics['correlation']
                self._save_checkpoint(epoch, val_metrics)
                
    def _validate_expression(self, val_loader):
        """Validate expression prediction"""
        self.model.eval()
        
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                expression = batch['expression'].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                hidden_states = outputs.last_hidden_state
                pooled = hidden_states.mean(dim=1)
                pred_expression = self.regressor(pooled).squeeze()
                
                all_preds.extend(pred_expression.cpu().numpy())
                all_true.extend(expression.cpu().numpy())
                
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(all_preds, all_true)
        mse = np.mean((all_preds - all_true)**2)
        
        return {
            'correlation': correlation,
            'mse': mse
        }
        
    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'metrics': metrics,
            'config': {
                'model_name': self.model_name,
                'task': self.task,
                'use_lora': self.use_lora
            }
        }
        
        # Save classifier/regressor if exists
        if hasattr(self, 'classifier'):
            checkpoint['classifier_state'] = self.classifier.state_dict()
        if hasattr(self, 'regressor'):
            checkpoint['regressor_state'] = self.regressor.state_dict()
            
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best model
        best_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        
        # Save LoRA weights separately if using LoRA
        if self.use_lora:
            self.model.save_pretrained(self.output_dir / "lora_weights")

# ======================== Data Loading Functions ========================

def load_regulatory_training_data(data_file: str, tokenizer):
    """Load training data for regulatory element detection"""
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    sequences = []
    labels = []
    
    for entry in data:
        sequences.append(entry['sequence'])
        labels.append({
            'elements': entry['regulatory_elements']
        })
        
    # Split train/val
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = RegulatoryElementDataset(
        train_seqs, train_labels, tokenizer
    )
    
    val_dataset = RegulatoryElementDataset(
        val_seqs, val_labels, tokenizer
    )
    
    return train_dataset, val_dataset

def load_expression_training_data(data_file: str, tokenizer):
    """Load training data for expression prediction"""
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    gene_sequences = []
    expression_values = []
    
    for entry in data:
        gene_sequences.append({
            'promoter': entry['promoter_sequence'],
            'gene_body': entry['gene_body_sequence']
        })
        expression_values.append(entry['expression_value'])
        
    # Split train/val
    train_genes, val_genes, train_expr, val_expr = train_test_split(
        gene_sequences, expression_values, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = ExpressionDataset(
        train_genes, train_expr, tokenizer
    )
    
    val_dataset = ExpressionDataset(
        val_genes, val_expr, tokenizer
    )
    
    return train_dataset, val_dataset

# ======================== Main Training Script ========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BRIDGE Model Training")
    parser.add_argument("--model", required=True, 
                       choices=['dnabert2', 'hyenadna', 'lucaone'],
                       help="Model to fine-tune")
    parser.add_argument("--task", required=True,
                       choices=['regulatory', 'expression'],
                       help="Task to train for")
    parser.add_argument("--data", required=True,
                       help="Training data file")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA for efficient fine-tuning")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    if args.model == 'dnabert2':
        from bridge_models import DNABert2Wrapper
        model_wrapper = DNABert2Wrapper()
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
    elif args.model == 'hyenadna':
        from bridge_models import HyenaDNAWrapper
        model_wrapper = HyenaDNAWrapper()
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
    elif args.model == 'lucaone':
        from bridge_models import LucaOneWrapper
        model_wrapper = LucaOneWrapper()
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        
    # Create trainer
    trainer = BridgeTrainer(
        model_wrapper,
        args.model,
        args.task,
        args.output,
        use_lora=args.use_lora
    )
    
    # Load data
    if args.task == 'regulatory':
        train_dataset, val_dataset = load_regulatory_training_data(
            args.data, tokenizer
        )
        
        # Train
        trainer.train_regulatory_detection(
            train_dataset,
            val_dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
    elif args.task == 'expression':
        train_dataset, val_dataset = load_expression_training_data(
            args.data, tokenizer
        )
        
        # Train
        trainer.train_expression_prediction(
            train_dataset,
            val_dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
    print(f"Training complete. Model saved to {args.output}")

if __name__ == "__main__":
    main()