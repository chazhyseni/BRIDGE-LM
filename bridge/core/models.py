"""
BRIDGE Model Wrappers - Unified interface for all biological models
Handles model loading, inference, and format conversion
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# ======================== Data Structures ========================

@dataclass
class DNASequence:
    """Standard DNA sequence representation"""
    sequence: str
    chromosome: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    strand: Optional[str] = '+'
    
    def __len__(self):
        return len(self.sequence)
    
    def get_subsequence(self, start: int, end: int) -> 'DNASequence':
        """Extract subsequence with proper coordinate tracking"""
        return DNASequence(
            sequence=self.sequence[start:end],
            chromosome=self.chromosome,
            start=self.start + start if self.start else start,
            end=self.start + end if self.start else end,
            strand=self.strand
        )

@dataclass 
class RegulatoryElement:
    """Detected regulatory element with metadata"""
    sequence: str
    chr: str
    start: int
    end: int
    element_type: str  # 'enhancer', 'promoter', 'silencer'
    score: float
    strand: str = '*'
    core_start: Optional[int] = None  # High-confidence core
    core_end: Optional[int] = None
    tf_motifs: List[Dict] = None
    conservation_score: Optional[float] = None
    target_genes: List[str] = None
    
    def __post_init__(self):
        if self.tf_motifs is None:
            self.tf_motifs = []
        if self.target_genes is None:
            self.target_genes = []
            
    @property
    def length(self):
        return self.end - self.start
        
    @property
    def core_length(self):
        if self.core_start and self.core_end:
            return self.core_end - self.core_start
        return self.length

# ======================== Base Model Wrapper ========================

class BaseModelWrapper:
    """Base class for all model wrappers"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        
    def load_model(self):
        """Override in subclasses"""
        raise NotImplementedError
        
    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """Batch prediction interface"""
        raise NotImplementedError
        
    def get_attention_maps(self, sequences: List[str]) -> List[np.ndarray]:
        """Extract attention patterns for boundary detection"""
        raise NotImplementedError

# ======================== HyenaDNA Wrapper ========================

class HyenaDNAWrapper(BaseModelWrapper):
    """
    HyenaDNA for long-range regulatory detection
    Handles sequences up to 450k bp
    """
    
    def __init__(self, model_path='LongSafari/hyenadna-medium-450k-seqlen', 
                 device='cuda', max_length=450000):
        super().__init__(device)
        self.model_path = model_path
        self.max_length = max_length
        self.load_model()
        
    def load_model(self):
        """Load HyenaDNA model"""
        logging.info(f"Loading HyenaDNA from {self.model_path}")
        
        # Import HyenaDNA components
        from transformers import AutoConfig
        
        # Load model config
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Initialize model
        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True
        ).to(self.device)
        
        # Character-level tokenizer for DNA
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.model.eval()
        
    def tokenize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Convert DNA sequences to token IDs"""
        max_len = min(self.max_length, max(len(seq) for seq in sequences))
        
        # Tokenize sequences
        input_ids = []
        attention_masks = []
        
        for seq in sequences:
            # Convert to token IDs
            tokens = [self.vocab.get(base, self.vocab['N']) for base in seq.upper()]
            
            # Truncate or pad
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.vocab['N']] * (max_len - len(tokens))
                
            input_ids.append(tokens)
            
            # Create attention mask
            mask = [1] * min(len(seq), max_len) + [0] * (max_len - min(len(seq), max_len))
            attention_masks.append(mask)
            
        return {
            'input_ids': torch.tensor(input_ids, device=self.device),
            'attention_mask': torch.tensor(attention_masks, device=self.device)
        }
        
    def predict_batch(self, sequences: List[str], window_size=8192, 
                     stride=4096) -> List[Dict]:
        """
        Predict regulatory elements with sliding window
        Returns coarse predictions that need refinement
        """
        all_predictions = []
        
        for seq_idx, sequence in enumerate(sequences):
            seq_predictions = []
            
            # Sliding window approach for long sequences
            for start in range(0, len(sequence), stride):
                end = min(start + window_size, len(sequence))
                window = sequence[start:end]
                
                # Skip if window too small
                if len(window) < 100:
                    continue
                    
                # Tokenize and predict
                inputs = self.tokenize([window])
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    # Get hidden states from last layer
                    hidden_states = outputs.hidden_states[-1]
                    
                    # Average pool to get sequence representation
                    seq_repr = hidden_states.mean(dim=1)
                    
                    # Classification head (would be trained)
                    # For now, use similarity to learned prototypes
                    score = self._compute_regulatory_score(seq_repr)
                    
                seq_predictions.append({
                    'start': start,
                    'end': end,
                    'score': float(score),
                    'sequence': window
                })
                
            all_predictions.append(seq_predictions)
            
        return all_predictions
        
    def _compute_regulatory_score(self, representation):
        """
        Compute regulatory element score from representation
        In practice, this would be a trained classifier
        """
        # Placeholder - would be learned
        # High activation in certain dimensions indicates regulatory potential
        regulatory_dims = [10, 47, 129, 255, 341]  # Example dimensions
        score = representation[0, regulatory_dims].mean()
        return torch.sigmoid(score * 2)  # Scale and squash to [0,1]
        
    def get_attention_maps(self, sequences: List[str]) -> List[np.ndarray]:
        """
        Extract attention patterns for boundary detection
        HyenaDNA uses state-space models, so we extract gate activations
        """
        attention_maps = []
        
        for sequence in sequences:
            inputs = self.tokenize([sequence])
            
            with torch.no_grad():
                # Get intermediate activations
                outputs = self.model(**inputs, output_attentions=True)
                
                # Extract gating patterns (proxy for attention)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attn = outputs.attentions[-1]  # Last layer
                    attn_map = attn[0].mean(dim=0).cpu().numpy()  # Average over heads
                else:
                    # Fallback: use hidden state changes as proxy
                    hidden = outputs.hidden_states[-1][0]
                    diffs = torch.diff(hidden, dim=0)
                    attn_map = diffs.abs().mean(dim=-1).cpu().numpy()
                    
                attention_maps.append(attn_map)
                
        return attention_maps

# ======================== DNABert2 Wrapper ========================

class DNABert2Wrapper(BaseModelWrapper):
    """
    DNABert2 for fine-grained regulatory element detection
    Best for sequences 100-512bp
    """
    
    def __init__(self, model_path='zhihan1996/DNABERT-2-117M', 
                 device='cuda', max_length=512):
        super().__init__(device)
        self.model_path = model_path
        self.max_length = max_length
        self.load_model()
        
    def load_model(self):
        """Load DNABert2 with BPE tokenizer"""
        logging.info(f"Loading DNABert2 from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load for sequence classification
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_labels=3  # enhancer, promoter, other
        ).to(self.device)
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        ).to(self.device)
        
        self.model.eval()
        self.classifier.eval()
        
    def predict_batch(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Fine-grained prediction for regulatory elements
        Returns class probabilities and confidence scores
        """
        # Tokenize all sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        predictions = {
            'enhancer_probs': [],
            'promoter_probs': [],
            'other_probs': [],
            'embeddings': []
        }
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get embeddings
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            # Classification
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            
            predictions['enhancer_probs'] = probs[:, 0].cpu().numpy()
            predictions['promoter_probs'] = probs[:, 1].cpu().numpy()
            predictions['other_probs'] = probs[:, 2].cpu().numpy()
            predictions['embeddings'] = embeddings.cpu().numpy()
            
        return predictions
        
    def scan_sequence_precise(self, sequence: str, window=100, stride=10) -> Dict:
        """
        High-resolution scanning for precise boundaries
        """
        scores = {
            'enhancer': [],
            'promoter': [],
            'positions': []
        }
        
        # Batch all windows for efficiency
        windows = []
        positions = []
        
        for i in range(0, len(sequence) - window + 1, stride):
            windows.append(sequence[i:i + window])
            positions.append(i + window // 2)  # Center position
            
        # Process in batches
        batch_size = 256
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            preds = self.predict_batch(batch)
            
            scores['enhancer'].extend(preds['enhancer_probs'])
            scores['promoter'].extend(preds['promoter_probs'])
            
        scores['positions'] = positions
        
        # Convert to numpy arrays
        for key in ['enhancer', 'promoter']:
            scores[key] = np.array(scores[key])
            
        return scores
        
    def get_attention_maps(self, sequences: List[str]) -> List[np.ndarray]:
        """Extract attention maps for boundary detection"""
        inputs = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        attention_maps = []
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
            # Average attention across all layers and heads
            for i in range(len(sequences)):
                attn_stack = []
                for layer_attn in outputs.attentions:
                    # layer_attn shape: (batch, heads, seq_len, seq_len)
                    avg_attn = layer_attn[i].mean(dim=0)  # Average over heads
                    attn_stack.append(avg_attn)
                    
                # Average over layers
                final_attn = torch.stack(attn_stack).mean(dim=0)
                attention_maps.append(final_attn.cpu().numpy())
                
        return attention_maps

# ======================== LucaOne Wrapper ========================

class LucaOneWrapper(BaseModelWrapper):
    """
    LucaOne for multi-modal validation and central dogma
    """
    
    def __init__(self, model_path='Yuanfei/LucaOne', device='cuda'):
        super().__init__(device)
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load LucaOne unified model"""
        logging.info(f"Loading LucaOne from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.eval()
        
    def validate_regulatory_logic(self, element: RegulatoryElement, 
                                 gene_sequence: str,
                                 protein_sequence: Optional[str] = None) -> Dict:
        """
        Validate that regulatory element -> gene -> protein makes biological sense
        """
        validation_results = {
            'element_gene_compatibility': 0.0,
            'central_dogma_score': 0.0,
            'predicted_impact': 'unknown'
        }
        
        # Check element-gene relationship
        element_embedding = self._encode_sequence(element.sequence, 'DNA')
        gene_embedding = self._encode_sequence(gene_sequence, 'DNA')
        
        # Compute compatibility
        compatibility = torch.cosine_similarity(
            element_embedding,
            gene_embedding,
            dim=-1
        ).item()
        
        validation_results['element_gene_compatibility'] = compatibility
        
        # If protein provided, check central dogma
        if protein_sequence:
            # Verify DNA->Protein relationship
            cd_score = self._verify_central_dogma(gene_sequence, protein_sequence)
            validation_results['central_dogma_score'] = cd_score
            
        # Predict regulatory impact
        impact = self._predict_regulatory_impact(element_embedding, gene_embedding)
        validation_results['predicted_impact'] = impact
        
        return validation_results
        
    def _encode_sequence(self, sequence: str, modality: str) -> torch.Tensor:
        """Encode sequence with modality token"""
        # Add modality token
        if modality == 'DNA':
            sequence = f"<DNA>{sequence}"
        elif modality == 'PROTEIN':
            sequence = f"<PROTEIN>{sequence}"
            
        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            max_length=2048,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        return embedding
        
    def _verify_central_dogma(self, dna: str, protein: str) -> float:
        """
        Verify DNA-protein relationship
        Returns confidence score
        """
        # Encode both sequences
        dna_emb = self._encode_sequence(dna, 'DNA')
        prot_emb = self._encode_sequence(protein, 'PROTEIN')
        
        # LucaOne learns central dogma relationships
        # High similarity = valid relationship
        similarity = torch.cosine_similarity(dna_emb, prot_emb, dim=-1)
        
        # Convert to probability
        score = torch.sigmoid(similarity * 5).item()
        
        return score
        
    def _predict_regulatory_impact(self, element_emb: torch.Tensor,
                                  gene_emb: torch.Tensor) -> str:
        """Predict how element affects gene"""
        # Compute interaction features
        diff = element_emb - gene_emb
        prod = element_emb * gene_emb
        
        # Simple classifier (would be trained)
        combined = torch.cat([diff, prod], dim=-1)
        score = combined.mean().item()
        
        if score > 0.5:
            return "strong_activation"
        elif score > 0:
            return "weak_activation"
        elif score > -0.5:
            return "weak_repression"
        else:
            return "strong_repression"

# ======================== Model Manager ========================

class BridgeModelManager:
    """
    Manages all models and coordinates predictions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_models(self):
        """Initialize all required models"""
        # HyenaDNA for long-range
        if self.config.get('use_hyenadna', True):
            self.models['hyenadna'] = HyenaDNAWrapper(
                device=self.device,
                max_length=self.config.get('hyena_max_length', 450000)
            )
            
        # DNABert2 for fine-grained
        if self.config.get('use_dnabert2', True):
            self.models['dnabert2'] = DNABert2Wrapper(
                device=self.device,
                max_length=self.config.get('dnabert_max_length', 512)
            )
            
        # LucaOne for validation
        if self.config.get('use_lucaone', True):
            self.models['lucaone'] = LucaOneWrapper(device=self.device)
            
        logging.info(f"Initialized {len(self.models)} models")
        
    def get_model(self, model_name: str) -> BaseModelWrapper:
        """Get specific model wrapper"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized")
        return self.models[model_name]
        
    async def predict_regulatory_elements(self, sequence: DNASequence) -> List[RegulatoryElement]:
        """
        Main prediction pipeline
        Coordinates all models for precise detection
        """
        # Stage 1: Coarse detection with HyenaDNA
        coarse_predictions = await self._coarse_detection(sequence)
        
        # Stage 2: Refine boundaries with DNABert2
        refined_elements = await self._refine_boundaries(sequence, coarse_predictions)
        
        # Stage 3: Validate with LucaOne
        validated_elements = await self._validate_elements(refined_elements)
        
        return validated_elements
        
    async def _coarse_detection(self, sequence: DNASequence) -> List[Dict]:
        """Stage 1: Long-range detection"""
        if 'hyenadna' not in self.models:
            return []
            
        # Run HyenaDNA prediction
        predictions = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.models['hyenadna'].predict_batch,
            [sequence.sequence]
        )
        
        # Extract high-scoring regions
        high_score_regions = []
        for pred_list in predictions:
            for pred in pred_list:
                if pred['score'] > self.config.get('coarse_threshold', 0.7):
                    high_score_regions.append(pred)
                    
        return high_score_regions
        
    async def _refine_boundaries(self, sequence: DNASequence, 
                                coarse_predictions: List[Dict]) -> List[RegulatoryElement]:
        """Stage 2: Precise boundary detection"""
        if 'dnabert2' not in self.models:
            return []
            
        refined_elements = []
        
        for coarse_pred in coarse_predictions:
            # Extract region with padding
            start = max(0, coarse_pred['start'] - 200)
            end = min(len(sequence), coarse_pred['end'] + 200)
            
            region = sequence.get_subsequence(start, end)
            
            # High-resolution scan
            scores = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.models['dnabert2'].scan_sequence_precise,
                region.sequence
            )
            
            # Find precise boundaries
            elements = self._extract_elements_from_scores(scores, region)
            refined_elements.extend(elements)
            
        return refined_elements
        
    def _extract_elements_from_scores(self, scores: Dict, 
                                    region: DNASequence) -> List[RegulatoryElement]:
        """Extract elements from score profiles"""
        elements = []
        
        # Process each element type
        for element_type in ['enhancer', 'promoter']:
            if element_type not in scores:
                continue
                
            profile = scores[element_type]
            positions = scores['positions']
            
            # Smooth profile
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(profile, sigma=3)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(
                smoothed,
                height=self.config.get(f'{element_type}_threshold', 0.7),
                distance=50,  # Minimum 50bp between peaks
                prominence=0.1
            )
            
            # Extract elements around peaks
            for peak_idx in peaks:
                # Find boundaries where score drops
                left_idx = peak_idx
                right_idx = peak_idx
                
                threshold = smoothed[peak_idx] * 0.7  # 70% of peak
                
                # Extend left
                while left_idx > 0 and smoothed[left_idx] > threshold:
                    left_idx -= 1
                    
                # Extend right  
                while right_idx < len(smoothed) - 1 and smoothed[right_idx] > threshold:
                    right_idx += 1
                    
                # Convert indices to positions
                elem_start = region.start + positions[left_idx]
                elem_end = region.start + positions[right_idx]
                
                # Create element
                element = RegulatoryElement(
                    sequence=region.sequence[positions[left_idx]:positions[right_idx]],
                    chr=region.chromosome,
                    start=elem_start,
                    end=elem_end,
                    element_type=element_type,
                    score=float(smoothed[peak_idx]),
                    core_start=region.start + positions[peak_idx] - 20,
                    core_end=region.start + positions[peak_idx] + 20
                )
                
                elements.append(element)
                
        return elements
        
    async def _validate_elements(self, elements: List[RegulatoryElement]) -> List[RegulatoryElement]:
        """Stage 3: Biological validation"""
        if 'lucaone' not in self.models:
            return elements
            
        validated = []
        
        for element in elements:
            # Quick validation based on sequence features
            if self._passes_basic_validation(element):
                validated.append(element)
                
        return validated
        
    def _passes_basic_validation(self, element: RegulatoryElement) -> bool:
        """Basic biological validation"""
        # Size constraints
        if element.element_type == 'enhancer':
            if not (100 <= element.length <= 2000):
                return False
        elif element.element_type == 'promoter':
            if not (100 <= element.length <= 1000):
                return False
                
        # GC content
        gc_count = element.sequence.count('G') + element.sequence.count('C')
        gc_content = gc_count / len(element.sequence)
        
        if not (0.3 <= gc_content <= 0.7):
            return False
            
        # Complexity check (no long repeats)
        if self._has_low_complexity(element.sequence):
            return False
            
        return True
        
    def _has_low_complexity(self, sequence: str, k=10) -> bool:
        """Check for low complexity regions"""
        # Simple check for repeats
        for i in range(len(sequence) - k):
            kmer = sequence[i:i+k]
            if sequence.count(kmer) > 5:  # Same 10-mer appears >5 times
                return True
        return False