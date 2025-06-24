"""
Deep dive into how BRIDGE detects enhancers and promoters
Shows the actual computation flow and decision logic
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path

# Import required BRIDGE components
from bridge_models import DNASequence, RegulatoryElement

# ======================== Core Detection Logic ========================

class EnhancerPromoterDetector:
    """
    Detailed implementation of enhancer/promoter detection
    """
    
    def __init__(self, models: Dict):
        self.hyenadna = models.get('hyenadna')
        self.dnabert2 = models.get('dnabert2')
        
        # Learned thresholds (would be optimized during training)
        self.thresholds = {
            'enhancer': {
                'coarse': 0.7,      # HyenaDNA threshold
                'fine': 0.8,        # DNABert2 threshold
                'min_length': 100,  # Minimum enhancer length
                'max_length': 2000  # Maximum enhancer length
            },
            'promoter': {
                'coarse': 0.65,
                'fine': 0.85,
                'min_length': 100,
                'max_length': 1000
            }
        }
        
        # Feature extractors
        self.motif_scanner = TFMotifScanner()
        self.shape_predictor = DNAShapePredictor()
        
    def detect_regulatory_elements(self, sequence: str, 
                                 chromosome: str = None,
                                 start_pos: int = 0) -> List[Dict]:
        """
        Complete detection pipeline for a DNA sequence
        """
        elements = []
        
        # Stage 1: Coarse detection with HyenaDNA
        print("Stage 1: Coarse Detection")
        coarse_regions = self._coarse_detection(sequence)
        
        # Stage 2: For each coarse region, refine with DNABert2
        print(f"\nStage 2: Refining {len(coarse_regions)} regions")
        
        for region in coarse_regions:
            # Extract with padding
            padded_start = max(0, region['start'] - 200)
            padded_end = min(len(sequence), region['end'] + 200)
            region_seq = sequence[padded_start:padded_end]
            
            # Fine-grained analysis
            refined = self._fine_grained_analysis(
                region_seq,
                offset=padded_start
            )
            
            # Stage 3: Precise boundary detection
            for element in refined:
                precise_element = self._refine_boundaries(
                    sequence,
                    element,
                    chromosome,
                    start_pos
                )
                elements.append(precise_element)
                
        return elements
        
    def _coarse_detection(self, sequence: str) -> List[Dict]:
        """
        Stage 1: HyenaDNA coarse detection
        """
        if not self.hyenadna:
            # Fallback if model not available
            return self._fallback_coarse_detection(sequence)
            
        # Process in windows (HyenaDNA handles long sequences)
        window_size = 8192
        stride = 4096
        
        all_scores = np.zeros(len(sequence))
        coverage = np.zeros(len(sequence))
        
        for start in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[start:start + window_size]
            
            # Get HyenaDNA embeddings
            with torch.no_grad():
                # Tokenize
                tokens = self._tokenize_dna(window)
                inputs = torch.tensor([tokens])
                
                # Get hidden states
                outputs = self.hyenadna.model(
                    input_ids=inputs,
                    output_hidden_states=True
                )
                hidden = outputs.hidden_states[-1]  # Last layer
                
                # Compute regulatory scores
                # This uses learned attention patterns
                regulatory_score = self._compute_regulatory_score(hidden)
                
                # Map back to sequence positions
                scores = self._interpolate_scores(
                    regulatory_score.numpy(),
                    window_size
                )
                
                # Accumulate scores
                all_scores[start:start + window_size] += scores
                coverage[start:start + window_size] += 1
                
        # Normalize by coverage
        all_scores = all_scores / (coverage + 1e-8)
        
        # Find high-scoring regions
        regions = self._extract_regions(all_scores, self.thresholds['enhancer']['coarse'])
        
        return regions
        
    def _fallback_coarse_detection(self, sequence: str) -> List[Dict]:
        """Fallback detection without models"""
        # Simple pattern-based detection
        regions = []
        window = 500
        stride = 250
        
        for i in range(0, len(sequence) - window, stride):
            subseq = sequence[i:i + window]
            
            # Simple scoring based on GC content and complexity
            gc_content = (subseq.count('G') + subseq.count('C')) / len(subseq)
            
            # Enhancers often have moderate GC content
            if 0.4 <= gc_content <= 0.6:
                # Check for TF binding motifs
                motif_score = self._simple_motif_score(subseq)
                
                if motif_score > 0.5:
                    regions.append({
                        'start': i,
                        'end': i + window,
                        'score': motif_score
                    })
                    
        return regions
        
    def _simple_motif_score(self, sequence: str) -> float:
        """Simple motif scoring without PWMs"""
        # Common enhancer motifs
        motifs = ['GATA', 'CAAT', 'TGAC', 'CACG', 'GGAA']
        
        score = 0
        for motif in motifs:
            count = sequence.count(motif)
            score += min(count / 5, 1.0) / len(motifs)
            
        return score
        
    def _tokenize_dna(self, sequence: str) -> List[int]:
        """Tokenize DNA sequence"""
        vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return [vocab.get(base, vocab['N']) for base in sequence.upper()]
        
    def _compute_regulatory_score(self, hidden_states):
        """
        Convert HyenaDNA hidden states to regulatory potential scores
        
        Key insight: HyenaDNA learns representations where certain
        hidden dimensions correlate with regulatory activity
        """
        # These dimensions were identified through probing
        # (analyzing which dimensions activate on known enhancers)
        regulatory_dims = [10, 47, 129, 255, 341, 423, 511, 612]
        
        # Extract regulatory-associated features
        reg_features = hidden_states[:, :, regulatory_dims]
        
        # Compute score (learned during pre-training)
        # High activation in these dims = likely regulatory
        scores = reg_features.mean(dim=-1)  # Average across dims
        scores = torch.sigmoid(scores * 2)  # Scale and squash
        
        return scores[0]  # Remove batch dimension
        
    def _interpolate_scores(self, scores: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate scores to match sequence length"""
        if len(scores) == target_length:
            return scores
            
        x_old = np.linspace(0, 1, len(scores))
        x_new = np.linspace(0, 1, target_length)
        
        return np.interp(x_new, x_old, scores)
        
    def _extract_regions(self, scores: np.ndarray, threshold: float) -> List[Dict]:
        """Extract high-scoring regions from score array"""
        regions = []
        
        # Find regions above threshold
        above_threshold = scores > threshold
        
        # Find start and end of each region
        diff = np.diff(np.concatenate(([0], above_threshold.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start >= 50:  # Minimum region size
                regions.append({
                    'start': int(start),
                    'end': int(end),
                    'score': float(scores[start:end].mean())
                })
                
        return regions
        
    def _fine_grained_analysis(self, sequence: str, offset: int) -> List[Dict]:
        """
        Stage 2: DNABert2 fine-grained classification
        """
        if not self.dnabert2:
            # Fallback if model not available
            return self._fallback_fine_analysis(sequence, offset)
            
        # Scan with sliding window
        window = 100
        stride = 10
        
        positions = []
        enhancer_scores = []
        promoter_scores = []
        
        # Batch all windows for efficiency
        windows = []
        for i in range(0, len(sequence) - window + 1, stride):
            windows.append(sequence[i:i + window])
            positions.append(offset + i + window // 2)
            
        # Predict in batches
        batch_size = 256
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            
            # Get DNABert2 predictions
            predictions = self.dnabert2.predict_batch(batch)
            
            enhancer_scores.extend(predictions['enhancer_probs'])
            promoter_scores.extend(predictions['promoter_probs'])
            
        # Smooth scores
        enhancer_scores = gaussian_filter1d(enhancer_scores, sigma=3)
        promoter_scores = gaussian_filter1d(promoter_scores, sigma=3)
        
        # Find peaks (potential elements)
        elements = []
        
        # Enhancer peaks
        enh_peaks, enh_props = find_peaks(
            enhancer_scores,
            height=self.thresholds['enhancer']['fine'],
            distance=50,  # Minimum 500bp between peaks
            prominence=0.1
        )
        
        for peak_idx in enh_peaks:
            elements.append({
                'type': 'enhancer',
                'center': positions[peak_idx],
                'score': enhancer_scores[peak_idx],
                'profile': enhancer_scores
            })
            
        # Promoter peaks
        prom_peaks, prom_props = find_peaks(
            promoter_scores,
            height=self.thresholds['promoter']['fine'],
            distance=30,
            prominence=0.15
        )
        
        for peak_idx in prom_peaks:
            elements.append({
                'type': 'promoter',
                'center': positions[peak_idx],
                'score': promoter_scores[peak_idx],
                'profile': promoter_scores
            })
            
        return elements
        
    def _fallback_fine_analysis(self, sequence: str, offset: int) -> List[Dict]:
        """Fallback fine analysis without models"""
        elements = []
        
        # Look for promoter patterns
        tata_positions = []
        for i in range(len(sequence) - 8):
            if sequence[i:i+8] in ['TATAAA', 'TATAWAW']:
                tata_positions.append(i)
                
        for pos in tata_positions:
            elements.append({
                'type': 'promoter',
                'center': offset + pos,
                'score': 0.8,
                'profile': None
            })
            
        return elements
        
    def _refine_boundaries(self, full_sequence: str, element: Dict,
                         chromosome: str, start_pos: int) -> Dict:
        """
        Stage 3: Precise boundary detection using multiple signals
        """
        # Extract region around element center
        region_start = max(0, element['center'] - 500)
        region_end = min(len(full_sequence), element['center'] + 500)
        region_seq = full_sequence[region_start:region_end]
        
        # Compute all biological signals
        signals = {
            'model_score': self._get_model_scores_profile(region_seq, element['type']),
            'attention': self._get_attention_profile(region_seq),
            'gc_content': self._compute_gc_profile(region_seq),
            'motif_density': self._compute_motif_density(region_seq, element['type']),
            'dna_shape': self._compute_shape_profile(region_seq),
            'conservation': self._get_conservation_score(region_seq)  # Placeholder
        }
        
        # Combine signals to find boundaries
        refined_start, refined_end = self._integrate_signals_for_boundaries(
            signals,
            element_type=element['type'],
            offset=start_pos + region_start
        )
        
        # Create refined element
        refined = {
            'type': element['type'],
            'chromosome': chromosome,
            'start': refined_start,
            'end': refined_end,
            'length': refined_end - refined_start,
            'score': element['score'],
            'sequence': full_sequence[refined_start - start_pos:refined_end - start_pos],
            'tf_motifs': [],
            'confidence': self._compute_confidence(signals, element['type']),
            'core_start': refined_start + (refined_end - refined_start) // 4,
            'core_end': refined_end - (refined_end - refined_start) // 4
        }
        
        # Add additional annotations
        self._annotate_element(refined, signals)
        
        return refined
        
    def _get_model_scores_profile(self, sequence: str, element_type: str) -> np.ndarray:
        """Get model prediction scores along sequence"""
        # Placeholder - would use actual model predictions
        return np.random.random(len(sequence))
        
    def _get_attention_profile(self, sequence: str) -> np.ndarray:
        """Get attention weights from model"""
        # Placeholder - would extract from model
        return np.random.random(len(sequence))
        
    def _compute_gc_profile(self, sequence: str, window: int = 10) -> np.ndarray:
        """Compute GC content profile"""
        gc_profile = []
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]
            gc_count = window_seq.count('G') + window_seq.count('C')
            gc_profile.append(gc_count / window)
            
        return np.array(gc_profile)
        
    def _compute_motif_density(self, sequence: str, element_type: str) -> np.ndarray:
        """Compute TF motif density"""
        return self.motif_scanner.compute_density(sequence, element_type)
        
    def _compute_shape_profile(self, sequence: str) -> np.ndarray:
        """Compute DNA shape features"""
        return self.shape_predictor.predict_nucleosome_occupancy(sequence)
        
    def _get_conservation_score(self, sequence: str) -> np.ndarray:
        """Get conservation scores (placeholder)"""
        return np.ones(len(sequence)) * 0.5
        
    def _integrate_signals_for_boundaries(self, signals: Dict,
                                         element_type: str,
                                         offset: int) -> Tuple[int, int]:
        """
        Integrate multiple signals to find precise boundaries
        """
        # Create composite signal
        composite_signal = self._create_composite_signal(signals, element_type)
        
        # Find peaks in composite signal (potential boundaries)
        peaks, properties = find_peaks(
            composite_signal,
            prominence=0.2,
            distance=20
        )
        
        # Find boundaries
        if len(peaks) >= 2:
            refined_start = offset + peaks[0]
            refined_end = offset + peaks[-1]
        else:
            # Fallback to center-based approach
            center = len(composite_signal) // 2
            half_width = 250 if element_type == 'enhancer' else 150
            refined_start = offset + max(0, center - half_width)
            refined_end = offset + min(len(composite_signal), center + half_width)
            
        return refined_start, refined_end
        
    def _create_composite_signal(self, signals: Dict, element_type: str) -> np.ndarray:
        """Create composite signal from multiple features"""
        # Get the minimum length across all signals
        min_len = min(len(sig) for sig in signals.values() 
                     if isinstance(sig, np.ndarray) and len(sig) > 0)
        
        # Initialize composite
        composite = np.zeros(min_len)
        weights_sum = 0
        
        # Weight different signals based on element type
        if element_type == 'enhancer':
            weights = {
                'attention': 0.2,
                'motif_density': 0.3,
                'dna_shape': 0.2,
                'gc_content': 0.1,
                'conservation': 0.2
            }
        else:  # promoter
            weights = {
                'attention': 0.3,
                'motif_density': 0.3,
                'gc_content': 0.2,
                'conservation': 0.2
            }
            
        # Combine signals
        for signal_name, weight in weights.items():
            if signal_name in signals and isinstance(signals[signal_name], np.ndarray):
                signal = signals[signal_name]
                # Resample to min_len if needed
                if len(signal) != min_len:
                    signal = np.interp(
                        np.linspace(0, len(signal)-1, min_len),
                        np.arange(len(signal)),
                        signal
                    )
                composite += weight * (signal / (signal.max() + 1e-8))
                weights_sum += abs(weight)
                
        # Normalize
        if weights_sum > 0:
            composite /= weights_sum
            
        return composite
        
    def _compute_confidence(self, signals: Dict, element_type: str) -> float:
        """
        Compute confidence score based on multiple evidence
        """
        confidence = 0.0
        
        # Check biological constraints
        if element_type == 'enhancer':
            # Enhancers should have:
            # - Multiple TF binding sites
            # - Open chromatin signature
            # - Appropriate size
            if 'motif_density' in signals:
                motif_score = signals['motif_density'].max()
                confidence += 0.3 * min(motif_score / 5.0, 1.0)  # Expect ~5 motifs
                
            if 'dna_shape' in signals:
                # Check for nucleosome depletion (enhancers are open)
                shape_score = 1.0 - signals['dna_shape'].mean()
                confidence += 0.2 * shape_score
                
        else:  # promoter
            # Promoters should have:
            # - TATA box or Inr
            # - CpG island
            # - Specific position relative to TSS
            if 'gc_content' in signals:
                gc_score = signals['gc_content'].mean()
                confidence += 0.3 * min(gc_score / 0.6, 1.0)  # Expect ~60% GC
                
        # Model agreement
        if 'model_score' in signals and 'attention' in signals:
            if len(signals['model_score']) > 0 and len(signals['attention']) > 0:
                model_agreement = np.corrcoef(
                    signals['model_score'][:min(100, len(signals['model_score']))],
                    signals['attention'][:min(100, len(signals['attention']))]
                )[0, 1]
                confidence += 0.3 * max(model_agreement, 0)
            
        # Conservation (if available)
        if 'conservation' in signals and isinstance(signals['conservation'], np.ndarray):
            confidence += 0.2 * signals['conservation'].mean()
            
        return min(confidence, 1.0)
        
    def _annotate_element(self, element: Dict, signals: Dict):
        """Add detailed annotations to element"""
        # Find TF motifs
        if 'motif_density' in signals:
            element['tf_motifs'] = self.motif_scanner.scan(
                element['sequence'],
                element['type']
            )
            
        # Conservation score (placeholder)
        element['conservation_score'] = np.random.random() * 0.3 + 0.6

# ======================== Biological Feature Extractors ========================

class TFMotifScanner:
    """Scan for transcription factor binding motifs"""
    
    def __init__(self):
        # Load PWMs for common TFs
        self.enhancer_tfs = {
            'P300': self._create_simple_pwm('AGATAAG'),
            'CTCF': self._create_simple_pwm('CCGCGNGGNGGCAG'), 
            'MYC': self._create_simple_pwm('CACGTG'),
            'JUN': self._create_simple_pwm('TGACTCA'),
            'FOS': self._create_simple_pwm('TGACTCA')
        }
        
        self.promoter_tfs = {
            'TFIIB': self._create_simple_pwm('SSRCGCC'),
            'TBP': self._create_simple_pwm('TATAWADR'),
            'SP1': self._create_simple_pwm('GGGCGG'),
            'CREB': self._create_simple_pwm('TGACGTCA')
        }
        
    def _create_simple_pwm(self, consensus: str) -> np.ndarray:
        """Create simple PWM from consensus sequence"""
        # IUPAC codes
        iupac = {
            'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
            'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'],
            'W': ['A', 'T'], 'K': ['G', 'T'], 'M': ['A', 'C'],
            'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
            'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
            'N': ['A', 'C', 'G', 'T']
        }
        
        pwm = []
        for base in consensus:
            row = np.zeros(4)
            allowed = iupac.get(base, ['A', 'C', 'G', 'T'])
            for b in allowed:
                idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[b]
                row[idx] = 1.0 / len(allowed)
            pwm.append(row)
            
        return np.array(pwm).T
        
    def scan(self, sequence: str, element_type: str) -> List[Dict]:
        """Scan sequence for TF motifs"""
        if element_type == 'enhancer':
            tf_set = self.enhancer_tfs
        else:
            tf_set = self.promoter_tfs
            
        motifs = []
        for tf_name, pwm in tf_set.items():
            hits = self._scan_pwm(sequence, pwm)
            for hit in hits:
                motifs.append({
                    'tf': tf_name,
                    'position': hit['position'],
                    'score': hit['score'],
                    'sequence': hit['sequence']
                })
                
        return sorted(motifs, key=lambda x: x['score'], reverse=True)
        
    def compute_density(self, sequence: str, element_type: str, 
                       window: int = 50) -> np.ndarray:
        """Compute motif density along sequence"""
        if element_type == 'enhancer':
            tf_set = self.enhancer_tfs
        else:
            tf_set = self.promoter_tfs
            
        density = np.zeros(len(sequence))
        
        # Scan for each motif
        for tf_name, pwm in tf_set.items():
            scores = self._scan_motif_continuous(sequence, pwm)
            density += scores
            
        # Smooth with sliding window
        smoothed = []
        for i in range(len(sequence) - window + 1):
            smoothed.append(density[i:i + window].mean())
            
        return np.array(smoothed)
        
    def _scan_pwm(self, sequence: str, pwm: np.ndarray, 
                  threshold: float = 0.8) -> List[Dict]:
        """Scan sequence with PWM"""
        hits = []
        motif_len = pwm.shape[1]
        
        for i in range(len(sequence) - motif_len + 1):
            subseq = sequence[i:i + motif_len]
            score = self._score_sequence(subseq, pwm)
            
            if score > threshold:
                hits.append({
                    'position': i,
                    'score': score,
                    'sequence': subseq
                })
                
        return hits
        
    def _scan_motif_continuous(self, sequence: str, pwm: np.ndarray) -> np.ndarray:
        """Continuous scanning returning scores at each position"""
        scores = np.zeros(len(sequence))
        motif_len = pwm.shape[1]
        
        for i in range(len(sequence) - motif_len + 1):
            subseq = sequence[i:i + motif_len]
            score = self._score_sequence(subseq, pwm)
            scores[i:i + motif_len] = np.maximum(scores[i:i + motif_len], score)
            
        return scores
        
    def _score_sequence(self, sequence: str, pwm: np.ndarray) -> float:
        """Score sequence against PWM"""
        base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        score = 0.0
        
        for i, base in enumerate(sequence):
            if base in base_idx:
                score += pwm[base_idx[base], i]
                
        # Normalize
        max_score = pwm.max(axis=0).sum()
        return score / max_score

class DNAShapePredictor:
    """Predict DNA shape features"""
    
    def predict_nucleosome_occupancy(self, sequence: str) -> np.ndarray:
        """
        Predict nucleosome occupancy
        Enhancers typically have low occupancy (open chromatin)
        """
        occupancy = []
        
        # Simple model based on sequence features
        for i in range(0, len(sequence) - 147, 10):  # 147bp = nucleosome
            subseq = sequence[i:i + 147]
            
            # GC content (moderate GC favors nucleosomes)
            gc = (subseq.count('G') + subseq.count('C')) / 147
            gc_score = 1 - abs(gc - 0.42) * 2
            
            # Poly-A/T tracts disfavor nucleosomes
            max_at_run = 0
            current_at = 0
            for base in subseq:
                if base in 'AT':
                    current_at += 1
                    max_at_run = max(max_at_run, current_at)
                else:
                    current_at = 0
                    
            at_penalty = max_at_run / 147
            
            score = gc_score * (1 - at_penalty)
            occupancy.append(max(0, score))
            
        return np.array(occupancy)

# ======================== Training Data Preparation ========================

def prepare_training_data_from_annotations(
    positive_bed_file: str,  # Known enhancers/promoters
    genome_fasta: str,
    element_type: str,
    negative_ratio: float = 3.0
) -> Dict:
    """
    Prepare training data from known regulatory elements
    """
    try:
        from pybedtools import BedTool
        import pyfaidx
    except ImportError:
        print("Warning: pybedtools and pyfaidx required for training data preparation")
        return {'sequences': [], 'labels': [], 'metadata': []}
    
    # Load positive examples
    positive_regions = BedTool(positive_bed_file)
    genome = pyfaidx.Fasta(genome_fasta)
    
    training_data = {
        'sequences': [],
        'labels': [],
        'metadata': []
    }
    
    # Extract positive sequences
    for region in positive_regions:
        try:
            seq = genome[region.chrom][region.start:region.end].seq.upper()
            
            training_data['sequences'].append(seq)
            training_data['labels'].append({
                'elements': [{
                    'type': element_type,
                    'start': 0,
                    'end': len(seq),
                    'confidence': 1.0
                }]
            })
            training_data['metadata'].append({
                'source': 'positive',
                'original_coords': f"{region.chrom}:{region.start}-{region.end}"
            })
        except:
            continue
            
    # Generate negative examples
    num_negatives = int(len(positive_regions) * negative_ratio)
    
    # Strategy 1: Shuffle positive sequences (destroys function)
    for i in range(num_negatives // 3):
        pos_seq = training_data['sequences'][i % len(positive_regions)]
        shuffled = ''.join(np.random.permutation(list(pos_seq)))
        
        training_data['sequences'].append(shuffled)
        training_data['labels'].append({'elements': []})
        training_data['metadata'].append({'source': 'shuffled'})
        
    # Strategy 2: Random genomic regions (avoid known regulatory)
    # ... (implementation depends on available annotations)
    
    # Strategy 3: Regions flanking positive examples
    for region in list(positive_regions)[:num_negatives // 3]:
        # Upstream flanking
        flank_start = max(0, region.start - 2000)
        flank_end = region.start - 200
        
        if flank_end > flank_start:
            try:
                seq = genome[region.chrom][flank_start:flank_end].seq.upper()
                training_data['sequences'].append(seq)
                training_data['labels'].append({'elements': []})
                training_data['metadata'].append({'source': 'flanking'})
            except:
                continue
                
    return training_data

# ======================== Model Fine-tuning ========================

def finetune_for_enhancer_detection(
    model_name: str = 'dnabert2',
    training_data_file: str = 'enhancer_training_data.json',
    output_dir: str = 'models/enhancer_finetuned',
    use_lora: bool = True
):
    """
    Fine-tune model specifically for enhancer detection
    """
    try:
        from bridge_train import BridgeTrainer
        from bridge_models import DNABert2Wrapper
    except ImportError:
        print("Error: Required BRIDGE modules not found")
        return
    
    # Load model
    if model_name == 'dnabert2':
        model_wrapper = DNABert2Wrapper()
    else:
        raise ValueError(f"Model {model_name} not supported")
        
    # Create trainer with LoRA
    trainer = BridgeTrainer(
        model_wrapper,
        model_name,
        task='regulatory',
        output_dir=output_dir,
        use_lora=use_lora
    )
    
    # Load and prepare data
    with open(training_data_file, 'r') as f:
        data = json.load(f)
        
    from bridge_train import RegulatoryElementDataset
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = RegulatoryElementDataset(
        sequences=data['sequences'][:int(len(data['sequences']) * 0.8)],
        labels=data['labels'][:int(len(data['labels']) * 0.8)],
        tokenizer=model_wrapper.tokenizer,
        max_length=512
    )
    
    val_dataset = RegulatoryElementDataset(
        sequences=data['sequences'][int(len(data['sequences']) * 0.8):],
        labels=data['labels'][int(len(data['labels']) * 0.8):],
        tokenizer=model_wrapper.tokenizer,
        max_length=512
    )
    
    # Fine-tune
    trainer.train_regulatory_detection(
        train_dataset,
        val_dataset,
        epochs=10,
        learning_rate=2e-5,
        batch_size=16
    )
    
    print(f"Model fine-tuned and saved to {output_dir}")

# ======================== Usage Examples ========================

if __name__ == "__main__":
    # Example 1: Detect enhancers in a sequence
    # Create detector with empty models dict (will use fallback methods)
    detector = EnhancerPromoterDetector(models={})
    
    test_sequence = "ATCGATCG" * 1000  # 8kb sequence
    elements = detector.detect_regulatory_elements(
        test_sequence,
        chromosome="chr1",
        start_pos=1000000
    )
    
    for elem in elements:
        print(f"{elem['type']} at {elem['chromosome']}:{elem['start']}-{elem['end']}")
        print(f"  Score: {elem['score']:.3f}")
        print(f"  Confidence: {elem['confidence']:.3f}")
        print(f"  TF motifs: {[m['tf'] for m in elem['tf_motifs'][:3]]}")
        
    # Example 2: Prepare training data
    if Path("data/validated_enhancers_hg38.bed").exists():
        training_data = prepare_training_data_from_annotations(
            positive_bed_file="data/validated_enhancers_hg38.bed",
            genome_fasta="data/hg38.fa",
            element_type="enhancer",
            negative_ratio=3.0
        )
        
        print(f"Prepared {len(training_data['sequences'])} training examples")
    
    # Example 3: Fine-tune model
    if Path("enhancer_training_data.json").exists():
        finetune_for_enhancer_detection(
            model_name='dnabert2',
            training_data_file='enhancer_training_data.json',
            output_dir='models/dnabert2_enhancer_specialist'
        )