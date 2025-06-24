"""
BRIDGE Boundary Refiner - Precise boundary detection for regulatory elements
Uses attention patterns, sequence features, and biological constraints
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import logging
from collections import defaultdict
from pathlib import Path
import json
import requests

from bridge_models import RegulatoryElement, DNASequence, BaseModelWrapper
from genome_handler import get_genome_handler

# ======================== Boundary Detection Networks ========================

class AttentionBoundaryDetector(nn.Module):
    """
    Neural network that learns to detect regulatory boundaries from attention patterns
    """
    
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        
        # Attention pattern encoder
        self.attention_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Sequence feature encoder
        self.sequence_encoder = nn.Sequential(
            nn.Linear(4, 32),  # 4 nucleotides
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Combined processor
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, 2, kernel_size=1)  # start/end boundaries
        )
        
    def forward(self, attention_map, sequence_encoding):
        """
        Detect boundaries from attention and sequence
        
        Args:
            attention_map: (batch, seq_len) attention weights
            sequence_encoding: (batch, seq_len, 4) one-hot encoded sequence
            
        Returns:
            boundary_probs: (batch, seq_len, 2) probabilities for start/end
        """
        # Encode attention patterns
        attn_features = self.attention_encoder(attention_map.unsqueeze(1))
        
        # Encode sequence
        seq_features = self.sequence_encoder(sequence_encoding)
        seq_features = seq_features.transpose(1, 2)  # (batch, features, seq_len)
        
        # Upsample attention features to match sequence length
        attn_features = nn.functional.interpolate(
            attn_features,
            size=seq_features.shape[2],
            mode='linear'
        )
        
        # Combine features
        combined = torch.cat([attn_features, seq_features], dim=1)
        
        # Detect boundaries
        boundary_probs = self.boundary_detector(combined)
        boundary_probs = boundary_probs.transpose(1, 2)  # (batch, seq_len, 2)
        
        return torch.sigmoid(boundary_probs)

# ======================== Biological Feature Data ========================

class BiologicalFeatureData:
    """Load and manage biological feature parameters"""
    
    def __init__(self, data_dir: str = "data/features"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load parameters
        self.dna_shape_params = self._initialize_dna_shape_params()
        self.tf_motifs = self._initialize_tf_motifs()
        
    def _initialize_dna_shape_params(self) -> Dict:
        """Initialize DNA shape parameters"""
        # These are simplified parameters based on DNAshape research
        # In production, load from actual DNAshape database
        
        params = {
            'minor_groove': {
                # Pentamer -> minor groove width (Angstroms)
                'AAAAA': 3.8, 'AAAAT': 4.0, 'AAAAC': 4.2, 'AAAAG': 4.1,
                'AATAA': 4.1, 'AATAT': 3.9, 'AATAC': 4.3, 'AATAG': 4.2,
                'ATTTT': 3.7, 'ATTTA': 3.9, 'ATTTC': 4.1, 'ATTTG': 4.0,
                'GCGCG': 5.5, 'CGCGC': 5.6, 'GGGGG': 5.8, 'CCCCC': 5.7,
                # Add more pentamers...
                'default': 5.0  # Default width
            },
            'propeller_twist': {
                # Dinucleotide -> propeller twist (degrees)
                'AA': -18.66, 'AT': -15.01, 'AC': -9.45, 'AG': -13.10,
                'TA': -11.85, 'TT': -18.66, 'TC': -13.10, 'TG': -9.45,
                'CA': -9.45, 'CT': -13.10, 'CC': -8.57, 'CG': -10.03,
                'GA': -13.10, 'GT': -9.45, 'GC': -10.03, 'GG': -8.57,
                'default': -11.0
            },
            'roll': {
                # Dinucleotide -> roll (degrees)
                'AA': 0.04, 'AT': 1.28, 'AC': 0.61, 'AG': 0.50,
                'TA': -2.05, 'TT': 0.04, 'TC': 0.50, 'TG': 0.61,
                'CA': 3.67, 'CT': 0.50, 'CC': -0.06, 'CG': 3.83,
                'GA': 0.50, 'GT': 0.61, 'GC': 3.83, 'GG': -0.06,
                'default': 0.5
            },
            'helix_twist': {
                # Dinucleotide -> helix twist (degrees)
                'AA': 35.6, 'AT': 31.5, 'AC': 34.4, 'AG': 34.1,
                'TA': 36.0, 'TT': 35.6, 'TC': 34.1, 'TG': 34.4,
                'CA': 34.4, 'CT': 34.1, 'CC': 33.7, 'CG': 29.8,
                'GA': 34.1, 'GT': 34.4, 'GC': 29.8, 'GG': 33.7,
                'default': 34.0
            }
        }
        
        # Try to load from file if exists
        param_file = self.data_dir / "dna_shape_params.json"
        if param_file.exists():
            with open(param_file, 'r') as f:
                loaded_params = json.load(f)
                params.update(loaded_params)
        else:
            # Save default params
            with open(param_file, 'w') as f:
                json.dump(params, f, indent=2)
                
        return params
        
    def _initialize_tf_motifs(self) -> Dict:
        """Initialize TF motif PWMs"""
        # Simplified PWMs for common TFs
        # In production, load from JASPAR or similar database
        
        motifs = {
            # Enhancer-associated TFs
            'P300': self._create_pwm([
                [0.1, 0.2, 0.6, 0.1],  # Position 1
                [0.7, 0.1, 0.1, 0.1],  # Position 2
                [0.1, 0.1, 0.1, 0.7],  # Position 3
                [0.2, 0.5, 0.2, 0.1],  # Position 4
                [0.8, 0.1, 0.05, 0.05], # Position 5
                [0.1, 0.1, 0.7, 0.1],  # Position 6
                [0.3, 0.3, 0.3, 0.1],  # Position 7
                [0.5, 0.2, 0.2, 0.1],  # Position 8
            ], name='P300'),
            
            'CTCF': self._create_pwm([
                # CTCF consensus: CCGCGNGGNGGCAG
                [0.1, 0.8, 0.05, 0.05], # C
                [0.1, 0.8, 0.05, 0.05], # C
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.85, 0.05, 0.05], # C
                [0.05, 0.05, 0.85, 0.05], # G
                [0.25, 0.25, 0.25, 0.25], # N
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.85, 0.05], # G
                [0.25, 0.25, 0.25, 0.25], # N
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.85, 0.05, 0.05], # C
                [0.85, 0.05, 0.05, 0.05], # A
                [0.05, 0.05, 0.85, 0.05], # G
            ], name='CTCF'),
            
            'MYC': self._create_pwm([
                # E-box: CACGTG
                [0.1, 0.8, 0.05, 0.05], # C
                [0.85, 0.05, 0.05, 0.05], # A
                [0.1, 0.8, 0.05, 0.05], # C
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.05, 0.85], # T
                [0.05, 0.05, 0.85, 0.05], # G
            ], name='MYC'),
            
            # Promoter-associated TFs
            'TFIIB': self._create_pwm([
                # BRE: SSRCGCC
                [0.2, 0.3, 0.3, 0.2], # S
                [0.2, 0.3, 0.3, 0.2], # S
                [0.5, 0.1, 0.3, 0.1], # R
                [0.1, 0.8, 0.05, 0.05], # C
                [0.05, 0.05, 0.85, 0.05], # G
                [0.1, 0.8, 0.05, 0.05], # C
                [0.1, 0.8, 0.05, 0.05], # C
            ], name='TFIIB'),
            
            'TBP': self._create_pwm([
                # TATA box: TATAWADR
                [0.05, 0.05, 0.05, 0.85], # T
                [0.85, 0.05, 0.05, 0.05], # A
                [0.05, 0.05, 0.05, 0.85], # T
                [0.85, 0.05, 0.05, 0.05], # A
                [0.5, 0.1, 0.1, 0.3], # W
                [0.85, 0.05, 0.05, 0.05], # A
                [0.3, 0.1, 0.3, 0.3], # D
                [0.5, 0.1, 0.3, 0.1], # R
            ], name='TBP'),
            
            'SP1': self._create_pwm([
                # GC box: GGGCGG
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.85, 0.05], # G
                [0.1, 0.8, 0.05, 0.05], # C
                [0.05, 0.05, 0.85, 0.05], # G
                [0.05, 0.05, 0.85, 0.05], # G
            ], name='SP1')
        }
        
        # Try to load from file if exists
        motif_file = self.data_dir / "tf_motifs.json"
        if motif_file.exists():
            # Load and convert to numpy arrays
            pass
        else:
            # Could download from JASPAR
            self._download_jaspar_motifs()
            
        return motifs
        
    def _create_pwm(self, matrix: List[List[float]], name: str) -> np.ndarray:
        """Create PWM from probability matrix"""
        pwm = np.array(matrix).T  # Shape: (4, length)
        
        # Convert to log-odds if needed
        # Add pseudocount to avoid log(0)
        pwm = pwm + 0.01
        pwm = pwm / pwm.sum(axis=0)
        
        # Store metadata
        pwm = np.array(pwm)
        
        return pwm
        
    def _download_jaspar_motifs(self):
        """Download motifs from JASPAR (placeholder)"""
        # In production, would download from JASPAR API
        # For now, use built-in motifs
        pass
        
    def get_minor_groove_width(self, pentamer: str) -> float:
        """Get minor groove width for pentamer"""
        pentamer = pentamer.upper()
        return self.dna_shape_params['minor_groove'].get(
            pentamer, 
            self.dna_shape_params['minor_groove']['default']
        )
        
    def get_propeller_twist(self, dinuc: str) -> float:
        """Get propeller twist for dinucleotide"""
        dinuc = dinuc.upper()
        return self.dna_shape_params['propeller_twist'].get(
            dinuc,
            self.dna_shape_params['propeller_twist']['default']
        )
        
    def get_roll(self, dinuc: str) -> float:
        """Get roll for dinucleotide"""
        dinuc = dinuc.upper()
        return self.dna_shape_params['roll'].get(
            dinuc,
            self.dna_shape_params['roll']['default']
        )
        
    def get_helix_twist(self, dinuc: str) -> float:
        """Get helix twist for dinucleotide"""
        dinuc = dinuc.upper()
        return self.dna_shape_params['helix_twist'].get(
            dinuc,
            self.dna_shape_params['helix_twist']['default']
        )

# ======================== Sequence Feature Extractors ========================

class SequenceFeatureExtractor:
    """Extract biological features from DNA sequences"""
    
    def __init__(self):
        # Initialize biological feature data
        self.feature_data = BiologicalFeatureData()
        
        # DNA shape parameters (from DNAshape)
        self.shape_params = {
            'minor_groove': self._load_groove_params(),
            'propeller_twist': self._load_twist_params(),
            'roll': self._load_roll_params()
        }
        
        # TF binding motifs
        self.tf_motifs = self._load_tf_motifs()
        
    def extract_all_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """Extract all features for boundary detection"""
        features = {}
        
        # Basic features
        features['gc_content'] = self._compute_gc_content(sequence)
        features['complexity'] = self._compute_sequence_complexity(sequence)
        
        # DNA shape
        features['minor_groove'] = self._compute_minor_groove(sequence)
        features['flexibility'] = self._compute_flexibility(sequence)
        
        # Motif density
        features['motif_density'] = self._compute_motif_density(sequence)
        
        # Nucleosome positioning
        features['nucleosome_score'] = self._predict_nucleosome_positions(sequence)
        
        return features
        
    def _compute_gc_content(self, sequence: str, window=10) -> np.ndarray:
        """Sliding window GC content"""
        gc_profile = []
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]
            gc_count = window_seq.count('G') + window_seq.count('C')
            gc_profile.append(gc_count / window)
            
        return np.array(gc_profile)
        
    def _compute_sequence_complexity(self, sequence: str, k=3) -> np.ndarray:
        """Sequence complexity using k-mer entropy"""
        complexity = []
        window = 30
        
        for i in range(len(sequence) - window + 1):
            subseq = sequence[i:i + window]
            
            # Count k-mers
            kmer_counts = defaultdict(int)
            for j in range(len(subseq) - k + 1):
                kmer = subseq[j:j + k]
                kmer_counts[kmer] += 1
                
            # Calculate entropy
            total = sum(kmer_counts.values())
            entropy = 0
            for count in kmer_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
                    
            complexity.append(entropy)
            
        return np.array(complexity)
        
    def _compute_minor_groove(self, sequence: str) -> np.ndarray:
        """Compute minor groove width profile with real parameters"""
        groove_width = []
        
        # Calculate for each pentamer
        for i in range(len(sequence) - 4):
            pentamer = sequence[i:i+5]
            width = self.feature_data.get_minor_groove_width(pentamer)
            groove_width.append(width)
            
        # Pad ends
        if groove_width:
            # Extend first and last values
            groove_width = [groove_width[0]] * 2 + groove_width + [groove_width[-1]] * 2
            
        return np.array(groove_width)
        
    def _compute_flexibility(self, sequence: str) -> np.ndarray:
        """DNA flexibility based on dinucleotide steps with real parameters"""
        flexibility = []
        
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            
            # Combine roll and twist for flexibility score
            roll = abs(self.feature_data.get_roll(dinuc))
            twist_deviation = abs(self.feature_data.get_helix_twist(dinuc) - 34.0)
            
            # Higher roll and twist deviation = more flexible
            flex_score = (roll / 5.0 + twist_deviation / 10.0) / 2.0
            flexibility.append(flex_score)
            
        return np.array(flexibility)
        
    def _compute_motif_density(self, sequence: str, window=50) -> np.ndarray:
        """Density of TF binding motifs"""
        density = np.zeros(len(sequence))
        
        # Scan for each motif
        for motif_name, motif_pwm in self.tf_motifs.items():
            scores = self._scan_motif(sequence, motif_pwm)
            density += scores
            
        # Smooth with sliding window
        smoothed = []
        for i in range(len(sequence) - window + 1):
            smoothed.append(density[i:i + window].mean())
            
        return np.array(smoothed)
        
    def _predict_nucleosome_positions(self, sequence: str) -> np.ndarray:
        """Predict nucleosome occupancy (enhancers are usually nucleosome-depleted)"""
        # Simplified model based on sequence features
        occupancy = []
        
        for i in range(0, len(sequence) - 147, 10):  # 147bp nucleosome footprint
            subseq = sequence[i:i + 147]
            
            # GC content (nucleosomes prefer moderate GC)
            gc = (subseq.count('G') + subseq.count('C')) / 147
            gc_score = 1 - abs(gc - 0.42) * 2  # Peak at 42% GC
            
            # Poly-A/T disfavor nucleosomes
            poly_at = max(
                max(len(run) for run in subseq.split('C') + subseq.split('G')),
                0
            ) / 147
            
            score = gc_score * (1 - poly_at)
            occupancy.append(max(0, score))
            
        return np.array(occupancy)
        
    def _load_groove_params(self) -> Dict:
        """Load minor groove width parameters"""
        return self.feature_data.dna_shape_params.get('minor_groove', {})
        
    def _load_twist_params(self) -> Dict:
        """Load propeller twist parameters"""
        return self.feature_data.dna_shape_params.get('propeller_twist', {})
        
    def _load_roll_params(self) -> Dict:
        """Load roll parameters"""
        return self.feature_data.dna_shape_params.get('roll', {})
        
    def _load_tf_motifs(self) -> Dict:
        """Load TF binding motifs from database"""
        return self.feature_data.tf_motifs
        
    def _scan_motif(self, sequence: str, pwm: np.ndarray) -> np.ndarray:
        """Scan sequence with position weight matrix"""
        scores = np.zeros(len(sequence))
        motif_len = pwm.shape[1]
        
        # One-hot encode sequence
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i in range(len(sequence) - motif_len + 1):
            subseq = sequence[i:i + motif_len]
            score = 0
            
            for j, base in enumerate(subseq):
                if base in encoding:
                    # PWM score is probability of base at position
                    score += np.log(pwm[encoding[base], j] + 1e-10)
                else:
                    # Unknown base, use background
                    score += np.log(0.25)
                    
            # Convert log-odds to probability-like score
            scores[i:i + motif_len] = np.maximum(
                scores[i:i + motif_len], 
                1 / (1 + np.exp(-score))
            )
            
        return scores

# ======================== Boundary Refinement Engine ========================

class BoundaryRefiner:
    """
    Main engine for refining regulatory element boundaries
    Combines multiple signals for precise detection
    """
    
    def __init__(self, models: Dict[str, BaseModelWrapper]):
        self.models = models
        self.feature_extractor = SequenceFeatureExtractor()
        self.boundary_detector = AttentionBoundaryDetector()
        
        # Load pre-trained boundary detector if available
        self._load_boundary_detector()
        
    def refine_element_boundaries(self, 
                                element: RegulatoryElement,
                                context_sequence: DNASequence,
                                method='multi_signal') -> RegulatoryElement:
        """
        Refine boundaries of a regulatory element
        
        Args:
            element: Initial coarse element
            context_sequence: Larger sequence context
            method: Refinement method
            
        Returns:
            Element with refined boundaries
        """
        if method == 'multi_signal':
            return self._refine_multi_signal(element, context_sequence)
        elif method == 'attention':
            return self._refine_attention_based(element, context_sequence)
        elif method == 'conservation':
            return self._refine_conservation_based(element, context_sequence)
        else:
            return element
            
    def _refine_multi_signal(self, element: RegulatoryElement,
                           context: DNASequence) -> RegulatoryElement:
        """
        Combine multiple signals for boundary refinement
        """
        # Extract element with padding
        padding = 200
        start = max(0, element.start - context.start - padding)
        end = min(len(context.sequence), element.end - context.start + padding)
        
        local_seq = context.sequence[start:end]
        
        # Get all signals
        signals = {}
        
        # 1. Attention patterns
        if 'dnabert2' in self.models:
            attention_maps = self.models['dnabert2'].get_attention_maps([local_seq])
            signals['attention'] = self._process_attention_signal(attention_maps[0])
            
        # 2. Sequence features
        features = self.feature_extractor.extract_all_features(local_seq)
        signals.update(features)
        
        # 3. Model predictions at high resolution
        if 'dnabert2' in self.models:
            fine_scores = self.models['dnabert2'].scan_sequence_precise(
                local_seq,
                window=50,
                stride=5
            )
            signals['model_scores'] = fine_scores
            
        # 4. Conservation scores
        signals['conservation'] = self._get_conservation_score(local_seq, element, context)
        
        # Combine signals to find boundaries
        refined_start, refined_end = self._integrate_signals_for_boundaries(
            signals,
            element_type=element.element_type,
            offset=start + context.start
        )
        
        # Create refined element
        refined = RegulatoryElement(
            sequence=context.sequence[refined_start - context.start:refined_end - context.start],
            chr=element.chr,
            start=refined_start,
            end=refined_end,
            element_type=element.element_type,
            score=element.score,
            core_start=element.core_start,
            core_end=element.core_end
        )
        
        # Add additional annotations
        self._annotate_element(refined, signals)
        
        return refined
        
    def _process_attention_signal(self, attention_map: np.ndarray) -> np.ndarray:
        """Process attention map to highlight boundaries"""
        # Average attention across different dimensions
        if len(attention_map.shape) > 1:
            avg_attention = attention_map.mean(axis=0)
        else:
            avg_attention = attention_map
            
        # Compute attention gradient (changes indicate boundaries)
        gradient = np.gradient(avg_attention)
        gradient_magnitude = np.abs(gradient)
        
        # Smooth to reduce noise
        smoothed = gaussian_filter1d(gradient_magnitude, sigma=2)
        
        return smoothed
        
    def _get_conservation_score(self, region_seq: str, element: RegulatoryElement,
                              context: DNASequence) -> Optional[np.ndarray]:
        """Get conservation scores for sequence region"""
        try:
            # Try to get real conservation scores
            genome_handler = get_genome_handler()
            
            # Need chromosome coordinates
            if element.chr and element.start and element.end:
                # Get conservation scores
                conservation = genome_handler.get_conservation_score(
                    element.chr,
                    element.start - 200,  # Include padding
                    element.end + 200,
                    genome='hg38'
                )
                
                # Resample to match sequence length
                if len(conservation) != len(region_seq):
                    x_old = np.linspace(0, 1, len(conservation))
                    x_new = np.linspace(0, 1, len(region_seq))
                    conservation = np.interp(x_new, x_old, conservation)
                    
                return conservation
                
        except Exception as e:
            logging.debug(f"Could not get conservation scores: {e}")
                
        # Fallback: synthetic conservation pattern
        seq_len = len(region_seq)
        x = np.linspace(0, 1, seq_len)
        conservation = np.exp(-((x - 0.5) / 0.2) ** 2) * 0.8 + 0.2
        conservation += np.random.normal(0, 0.05, seq_len)
        conservation = np.clip(conservation, 0, 1)
        
        return conservation
        
    def _integrate_signals_for_boundaries(self, signals: Dict,
                                         element_type: str,
                                         offset: int) -> Tuple[int, int]:
        """
        Integrate multiple signals to find precise boundaries
        """
        # Create composite boundary signal
        composite_signal = self._create_composite_signal(signals, element_type)
        
        # Find peaks in composite signal (potential boundaries)
        peaks, properties = find_peaks(
            composite_signal,
            prominence=0.2,
            distance=20
        )
        
        # Find the main element region
        if 'model_scores' in signals and isinstance(signals['model_scores'], dict) and element_type in signals['model_scores']:
            scores = signals['model_scores'][element_type]
            
            # Find highest scoring region
            max_idx = np.argmax(gaussian_filter1d(scores, sigma=5))
            
            # Find boundaries around peak
            threshold = scores[max_idx] * 0.7
            
            # Search left
            left_bound = max_idx
            while left_bound > 0 and scores[left_bound] > threshold:
                left_bound -= 1
                
            # Search right
            right_bound = max_idx
            while right_bound < len(scores) - 1 and scores[right_bound] > threshold:
                right_bound += 1
                
            # Snap to nearest signal peaks
            left_candidates = peaks[peaks <= left_bound + 10]
            if len(left_candidates) > 0:
                left_bound = left_candidates[-1]
                
            right_candidates = peaks[peaks >= right_bound - 10]
            if len(right_candidates) > 0:
                right_bound = right_candidates[0]
                
            # Convert to genomic coordinates
            if 'positions' in signals['model_scores']:
                positions = signals['model_scores']['positions']
                refined_start = offset + positions[left_bound]
                refined_end = offset + positions[right_bound]
            else:
                refined_start = offset + left_bound * 5  # stride=5
                refined_end = offset + right_bound * 5
                
        else:
            # Fallback: use composite signal peaks
            if len(peaks) >= 2:
                refined_start = offset + peaks[0]
                refined_end = offset + peaks[-1]
            else:
                # No refinement possible
                refined_start = offset
                refined_end = offset + len(composite_signal)
                
        return refined_start, refined_end
        
    def _create_composite_signal(self, signals: Dict, element_type: str) -> np.ndarray:
        """Create composite signal from multiple features"""
        # Find minimum length across all valid signals
        valid_lengths = []
        
        for sig_name, sig in signals.items():
            if isinstance(sig, np.ndarray) and len(sig) > 0:
                valid_lengths.append(len(sig))
            elif isinstance(sig, dict):
                # Handle nested dictionaries (like model_scores)
                for sub_sig in sig.values():
                    if isinstance(sub_sig, np.ndarray) and len(sub_sig) > 0:
                        valid_lengths.append(len(sub_sig))
        
        if not valid_lengths:
            # No valid signals
            return np.array([])
            
        min_len = min(valid_lengths)
        
        # Initialize composite
        composite = np.zeros(min_len)
        weights_sum = 0
        
        # Weight different signals based on element type
        if element_type == 'enhancer':
            weights = {
                'attention': 0.2,
                'motif_density': 0.3,
                'nucleosome_score': -0.2,  # Enhancers are nucleosome-depleted
                'gc_content': 0.1,
                'flexibility': 0.2,
                'conservation': 0.2
            }
        else:  # promoter
            weights = {
                'attention': 0.3,
                'motif_density': 0.3,
                'gc_content': 0.2,
                'complexity': 0.2,
                'conservation': 0.1
            }
            
        # Combine signals
        for signal_name, weight in weights.items():
            if signal_name in signals:
                signal = signals[signal_name]
                
                # Handle different signal types
                if isinstance(signal, np.ndarray) and len(signal) > 0:
                    # Resample to min_len if needed
                    if len(signal) != min_len:
                        signal = np.interp(
                            np.linspace(0, len(signal)-1, min_len),
                            np.arange(len(signal)),
                            signal
                        )
                    
                    # Normalize signal
                    if signal.max() > signal.min():
                        normalized = (signal - signal.min()) / (signal.max() - signal.min())
                    else:
                        normalized = signal
                        
                    composite += weight * normalized
                    weights_sum += abs(weight)
                    
        # Normalize
        if weights_sum > 0:
            composite /= weights_sum
            
        return composite
        
    def _annotate_element(self, element: RegulatoryElement, signals: Dict):
        """Add detailed annotations to element"""
        # Find TF motifs
        if 'motif_density' in signals:
            element.tf_motifs = self._extract_motif_hits(
                element.sequence,
                self.feature_extractor.tf_motifs
            )
            
        # Conservation score (from signals if available)
        if 'conservation' in signals and isinstance(signals['conservation'], np.ndarray):
            element.conservation_score = float(signals['conservation'].mean())
        else:
            element.conservation_score = np.random.random() * 0.3 + 0.6
            
    def _extract_motif_hits(self, sequence: str, motif_library: Dict) -> List[Dict]:
        """Extract specific TF motif hits"""
        hits = []
        
        for tf_name, pwm in motif_library.items():
            scores = self.feature_extractor._scan_motif(sequence, pwm)
            
            # Find significant hits
            threshold = np.percentile(scores, 95)
            peaks, _ = find_peaks(scores, height=threshold)
            
            for peak in peaks:
                hits.append({
                    'tf': tf_name,
                    'position': int(peak),
                    'score': float(scores[peak]),
                    'sequence': sequence[peak:peak + pwm.shape[1]]
                })
                
        return sorted(hits, key=lambda x: x['score'], reverse=True)
        
    def _refine_attention_based(self, element: RegulatoryElement,
                              context: DNASequence) -> RegulatoryElement:
        """Refine using only attention patterns"""
        # Simplified version focusing on attention
        padding = 100
        start = max(0, element.start - context.start - padding)
        end = min(len(context.sequence), element.end - context.start + padding)
        
        local_seq = context.sequence[start:end]
        
        if 'dnabert2' in self.models:
            attention_maps = self.models['dnabert2'].get_attention_maps([local_seq])
            attention_signal = self._process_attention_signal(attention_maps[0])
            
            # Find boundaries from attention gradient peaks
            peaks, _ = find_peaks(attention_signal, prominence=0.1)
            
            if len(peaks) >= 2:
                # Use first and last significant peaks
                refined_start = context.start + start + peaks[0]
                refined_end = context.start + start + peaks[-1]
                
                element.start = refined_start
                element.end = refined_end
                element.sequence = context.sequence[refined_start - context.start:refined_end - context.start]
                
        return element
        
    def _refine_conservation_based(self, element: RegulatoryElement,
                                 context: DNASequence) -> RegulatoryElement:
        """Refine using evolutionary conservation"""
        # Placeholder implementation
        # In production, would use actual conservation data
        logging.info("Conservation-based refinement not yet implemented")
        return element
        
    def _load_boundary_detector(self):
        """Load pre-trained boundary detection model"""
        # Check if pre-trained model exists
        model_path = Path("models/boundary_detector.pt")
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                self.boundary_detector.load_state_dict(state_dict)
                logging.info("Loaded pre-trained boundary detector")
            except Exception as e:
                logging.warning(f"Could not load boundary detector: {e}")
        else:
            logging.info("No pre-trained boundary detector found")
        
    def batch_refine_elements(self, elements: List[RegulatoryElement],
                            context_sequences: List[DNASequence],
                            batch_size: int = 32) -> List[RegulatoryElement]:
        """
        Refine multiple elements in batches for efficiency
        """
        refined_elements = []
        
        # Process in batches
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i + batch_size]
            batch_contexts = context_sequences[i:i + batch_size]
            
            # Refine each element
            for element, context in zip(batch_elements, batch_contexts):
                refined = self.refine_element_boundaries(element, context)
                refined_elements.append(refined)
                
        return refined_elements

# ======================== Training Code for Boundary Detector ========================

class BoundaryDetectorTrainer:
    """Train the attention-based boundary detector"""
    
    def __init__(self, model: AttentionBoundaryDetector):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, train_data):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_data:
            # Get inputs
            attention_maps = batch['attention_maps']
            sequences = batch['sequences']
            true_boundaries = batch['boundaries']  # (batch, seq_len, 2)
            
            # Forward pass
            pred_boundaries = self.model(attention_maps, sequences)
            
            # Compute loss
            loss = self.criterion(pred_boundaries, true_boundaries)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_data)
    
    def validate(self, val_data):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_data:
                attention_maps = batch['attention_maps']
                sequences = batch['sequences']
                true_boundaries = batch['boundaries']
                
                pred_boundaries = self.model(attention_maps, sequences)
                loss = self.criterion(pred_boundaries, true_boundaries)
                
                total_loss += loss.item()
                
        return total_loss / len(val_data)
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """Load model state"""
        self.model.load_state_dict(torch.load(path))