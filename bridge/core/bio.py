"""
BRIDGE Biological Validation - Connect regulatory elements to genes and validate predictions
Handles gene annotation, expression prediction, and central dogma validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import pybedtools
from collections import defaultdict

from bridge_models import RegulatoryElement, DNASequence, LucaOneWrapper

# ======================== Gene Data Structures ========================

@dataclass
class Gene:
    """Gene annotation with regulatory metadata"""
    gene_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    strand: str
    gene_type: str  # protein_coding, lncRNA, etc.
    
    # Regulatory features
    tss: int  # Transcription start site
    promoter_start: int
    promoter_end: int
    
    # Expression data
    expression_levels: Dict[str, float] = field(default_factory=dict)
    
    # Sequence data
    sequence: Optional[str] = None
    cds_start: Optional[int] = None
    cds_end: Optional[int] = None
    protein_sequence: Optional[str] = None
    
    # Regulatory connections
    regulatory_elements: List[RegulatoryElement] = field(default_factory=list)
    transcription_factors: List[str] = field(default_factory=list)
    
    @property
    def length(self):
        return self.end - self.start
        
    @property
    def promoter_sequence(self):
        if self.sequence and self.promoter_start and self.promoter_end:
            p_start = self.promoter_start - self.start
            p_end = self.promoter_end - self.start
            return self.sequence[p_start:p_end]
        return None

@dataclass
class RegulatoryInteraction:
    """Validated regulatory element-gene interaction"""
    element: RegulatoryElement
    gene: Gene
    distance: int  # Distance to TSS
    interaction_type: str  # 'promoter', 'proximal', 'distal'
    confidence: float
    validation_method: str  # How it was validated
    evidence: Dict = field(default_factory=dict)

# ======================== Gene Annotation Manager ========================

class GeneAnnotationManager:
    """
    Manages gene annotations and regulatory assignments
    """
    
    def __init__(self, annotation_file: str, genome: str = 'hg38'):
        self.genome = genome
        self.genes = {}
        self.chr_gene_trees = {}  # Interval trees for fast lookup
        
        # Load annotations
        self._load_annotations(annotation_file)
        self._build_interval_trees()
        
    def _load_annotations(self, annotation_file: str):
        """Load gene annotations from GTF/GFF file"""
        logging.info(f"Loading gene annotations from {annotation_file}")
        
        # Parse GTF file
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                    
                if parts[2] == 'gene':
                    # Parse gene entry
                    chrom = parts[0]
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[6]
                    
                    # Parse attributes
                    attrs = self._parse_gtf_attributes(parts[8])
                    
                    # Create gene object
                    gene = Gene(
                        gene_id=attrs.get('gene_id', ''),
                        gene_name=attrs.get('gene_name', attrs.get('gene_id', '')),
                        chromosome=chrom,
                        start=start,
                        end=end,
                        strand=strand,
                        gene_type=attrs.get('gene_type', 'unknown'),
                        tss=start if strand == '+' else end,
                        promoter_start=start - 2000 if strand == '+' else end - 500,
                        promoter_end=start + 500 if strand == '+' else end + 2000
                    )
                    
                    self.genes[gene.gene_id] = gene
                    
        logging.info(f"Loaded {len(self.genes)} genes")
        
    def _parse_gtf_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse GTF attribute string"""
        attrs = {}
        
        for attr in attr_string.strip().split(';'):
            attr = attr.strip()
            if not attr:
                continue
                
            if ' ' in attr:
                key, value = attr.split(' ', 1)
                attrs[key] = value.strip('"')
                
        return attrs
        
    def _build_interval_trees(self):
        """Build interval trees for efficient range queries"""
        from intervaltree import IntervalTree
        
        # Group genes by chromosome
        chr_genes = defaultdict(list)
        for gene in self.genes.values():
            chr_genes[gene.chromosome].append(gene)
            
        # Build interval tree for each chromosome
        for chrom, gene_list in chr_genes.items():
            tree = IntervalTree()
            
            for gene in gene_list:
                # Add gene region
                tree.addi(gene.start, gene.end, gene)
                
                # Add extended promoter region
                tree.addi(gene.promoter_start, gene.promoter_end, gene)
                
            self.chr_gene_trees[chrom] = tree
            
    def find_nearby_genes(self, element: RegulatoryElement,
                         max_distance: int = 1000000) -> List[Tuple[Gene, int]]:
        """
        Find genes near a regulatory element
        
        Returns:
            List of (gene, distance_to_tss) tuples
        """
        if element.chr not in self.chr_gene_trees:
            return []
            
        # Query interval tree
        tree = self.chr_gene_trees[element.chr]
        
        # Search region
        search_start = element.start - max_distance
        search_end = element.end + max_distance
        
        # Find overlapping genes
        overlapping = tree.overlap(search_start, search_end)
        
        # Calculate distances to TSS
        gene_distances = []
        
        for interval in overlapping:
            gene = interval.data
            
            # Calculate distance to TSS
            if gene.strand == '+':
                distance = element.start - gene.tss
            else:
                distance = gene.tss - element.end
                
            # Only include if within max distance
            if abs(distance) <= max_distance:
                gene_distances.append((gene, distance))
                
        # Sort by distance
        gene_distances.sort(key=lambda x: abs(x[1]))
        
        return gene_distances
        
    def assign_elements_to_genes(self, elements: List[RegulatoryElement],
                               assignment_rules: Dict = None) -> List[RegulatoryInteraction]:
        """
        Assign regulatory elements to their target genes
        """
        if assignment_rules is None:
            assignment_rules = {
                'max_distance': 1000000,  # 1Mb
                'promoter_distance': 2500,
                'proximal_distance': 50000,
                'prefer_closest': True,
                'allow_multiple': True
            }
            
        interactions = []
        
        for element in elements:
            # Find nearby genes
            nearby_genes = self.find_nearby_genes(
                element,
                max_distance=assignment_rules['max_distance']
            )
            
            if not nearby_genes:
                continue
                
            # Assign based on rules
            assigned_genes = []
            
            for gene, distance in nearby_genes:
                # Classify interaction type
                if abs(distance) <= assignment_rules['promoter_distance']:
                    interaction_type = 'promoter'
                    confidence = 0.9
                elif abs(distance) <= assignment_rules['proximal_distance']:
                    interaction_type = 'proximal'
                    confidence = 0.7
                else:
                    interaction_type = 'distal'
                    confidence = 0.5
                    
                # Adjust confidence based on element type
                if element.element_type == 'promoter' and interaction_type != 'promoter':
                    confidence *= 0.5  # Promoters should be near TSS
                    
                # Create interaction
                interaction = RegulatoryInteraction(
                    element=element,
                    gene=gene,
                    distance=distance,
                    interaction_type=interaction_type,
                    confidence=confidence,
                    validation_method='distance_based'
                )
                
                assigned_genes.append(interaction)
                
                # If only assigning to closest
                if assignment_rules['prefer_closest'] and not assignment_rules['allow_multiple']:
                    break
                    
            interactions.extend(assigned_genes)
            
        return interactions

# ======================== Expression Prediction ========================

class ExpressionPredictor:
    """
    Predicts gene expression from regulatory elements
    """
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
        # Expression prediction models
        self.expression_models = self._load_expression_models()
        
    def _load_expression_models(self):
        """Load pre-trained expression models"""
        # Placeholder - would load actual models
        return {}
        
    def predict_expression_from_elements(self, 
                                       gene: Gene,
                                       elements: List[RegulatoryElement],
                                       cell_type: str = 'generic') -> float:
        """
        Predict gene expression level from regulatory elements
        """
        # Extract features
        features = self._extract_expression_features(gene, elements, cell_type)
        
        # Apply model
        if cell_type in self.expression_models:
            model = self.expression_models[cell_type]
            expression = model.predict([features])[0]
        else:
            # Simple additive model
            expression = self._simple_expression_model(gene, elements)
            
        return expression
        
    def _extract_expression_features(self, gene: Gene, 
                                   elements: List[RegulatoryElement],
                                   cell_type: str) -> np.ndarray:
        """Extract features for expression prediction"""
        features = []
        
        # Promoter features
        promoter_elements = [e for e in elements if e.interaction_type == 'promoter']
        features.append(len(promoter_elements))
        features.append(max([e.score for e in promoter_elements]) if promoter_elements else 0)
        
        # Enhancer features
        enhancers = [e for e in elements if e.element_type == 'enhancer']
        features.append(len(enhancers))
        
        # Distance-weighted enhancer scores
        enhancer_contribution = 0
        for e in enhancers:
            distance = abs(e.distance)
            weight = np.exp(-distance / 50000)  # Exponential decay
            enhancer_contribution += e.score * weight
            
        features.append(enhancer_contribution)
        
        # Gene features
        features.append(1 if gene.gene_type == 'protein_coding' else 0)
        features.append(np.log10(gene.length + 1))
        
        return np.array(features)
        
    def _simple_expression_model(self, gene: Gene,
                               elements: List[RegulatoryElement]) -> float:
        """Simple additive model for expression"""
        base_expression = 1.0
        
        # Promoter contribution
        promoter_elements = [e for e in elements 
                           if abs(e.distance) < 2500]
        if promoter_elements:
            promoter_strength = max(e.score for e in promoter_elements)
            base_expression *= (1 + promoter_strength)
            
        # Enhancer contributions
        enhancer_contribution = 0
        for element in elements:
            if element.element_type == 'enhancer':
                # Distance decay
                distance = abs(element.distance)
                weight = np.exp(-distance / 100000)
                enhancer_contribution += element.score * weight
                
        # Final expression
        expression = base_expression * (1 + enhancer_contribution)
        
        return np.log2(expression + 1)  # Log scale

# ======================== Central Dogma Validator ========================

class CentralDogmaValidator:
    """
    Validates regulatory predictions using central dogma principles
    """
    
    def __init__(self, lucaone_model: LucaOneWrapper):
        self.lucaone = lucaone_model
        
    def validate_gene_regulation(self, 
                               gene: Gene,
                               regulatory_elements: List[RegulatoryElement],
                               expression_level: float) -> Dict:
        """
        Validate that regulatory elements -> expression -> protein makes sense
        """
        validation = {
            'regulatory_logic': 0.0,
            'expression_correlation': 0.0,
            'central_dogma_valid': False,
            'protein_production': 0.0,
            'feedback_score': 0.0
        }
        
        # Check regulatory logic
        validation['regulatory_logic'] = self._check_regulatory_logic(
            gene, regulatory_elements
        )
        
        # If protein-coding, validate central dogma
        if gene.gene_type == 'protein_coding' and gene.protein_sequence:
            cd_score = self.lucaone._verify_central_dogma(
                gene.sequence[gene.cds_start:gene.cds_end],
                gene.protein_sequence
            )
            validation['central_dogma_valid'] = cd_score > 0.8
            
            # Predict protein production
            validation['protein_production'] = self._predict_protein_level(
                expression_level,
                gene
            )
            
        # Check for TF feedback
        if gene.gene_name in self._get_tf_list():
            validation['feedback_score'] = self._check_tf_feedback(
                gene, regulatory_elements
            )
            
        return validation
        
    def _check_regulatory_logic(self, gene: Gene,
                              elements: List[RegulatoryElement]) -> float:
        """Check if regulatory architecture makes biological sense"""
        score = 0.0
        
        # Should have promoter elements
        promoter_elements = [e for e in elements 
                           if abs(e.distance) < 2500]
        if promoter_elements:
            score += 0.3
            
        # Enhancers should have appropriate distance distribution
        enhancers = [e for e in elements if e.element_type == 'enhancer']
        if enhancers:
            distances = [abs(e.distance) for e in enhancers]
            
            # Typical enhancer distance: 10-500kb
            typical_enhancers = sum(1 for d in distances 
                                  if 10000 < d < 500000)
            if typical_enhancers > 0:
                score += 0.3
                
        # Check for TF motifs in elements
        tf_motifs_found = sum(len(e.tf_motifs) for e in elements)
        if tf_motifs_found > 0:
            score += 0.2
            
        # Reasonable number of regulatory elements
        if 1 <= len(elements) <= 20:
            score += 0.2
            
        return min(score, 1.0)
        
    def _predict_protein_level(self, mrna_expression: float,
                              gene: Gene) -> float:
        """Predict protein level from mRNA expression"""
        # Simple model - would be more sophisticated
        
        # Translation efficiency factors
        efficiency = 1.0
        
        # 5' UTR length affects translation
        if gene.cds_start:
            utr_length = gene.cds_start - gene.start
            if utr_length < 100:
                efficiency *= 1.2  # Short UTR
            elif utr_length > 500:
                efficiency *= 0.8  # Long UTR
                
        # Codon optimization (simplified)
        if gene.sequence:
            cds = gene.sequence[gene.cds_start:gene.cds_end]
            gc_content = (cds.count('G') + cds.count('C')) / len(cds)
            
            # Optimal GC content ~50%
            efficiency *= 1 - abs(gc_content - 0.5)
            
        # Protein level proportional to mRNA with efficiency
        protein_level = mrna_expression * efficiency
        
        return protein_level
        
    def _check_tf_feedback(self, tf_gene: Gene,
                         regulatory_elements: List[RegulatoryElement]) -> float:
        """Check if TF regulates its own expression"""
        feedback_score = 0.0
        
        # Look for TF binding sites in regulatory elements
        tf_name = tf_gene.gene_name
        
        for element in regulatory_elements:
            for motif in element.tf_motifs:
                if motif['tf'] == tf_name:
                    # Found potential auto-regulation
                    feedback_score = max(feedback_score, motif['score'])
                    
        return feedback_score
        
    def _get_tf_list(self) -> Set[str]:
        """Get list of known transcription factors"""
        # Placeholder - would load from database
        return {'CTCF', 'P300', 'MYC', 'P53', 'NANOG', 'OCT4', 'SOX2'}

# ======================== Integrated Analysis ========================

class IntegratedBiologicalAnalysis:
    """
    Combines all biological validation and analysis components
    """
    
    def __init__(self, gene_manager: GeneAnnotationManager,
                 expression_predictor: ExpressionPredictor,
                 dogma_validator: CentralDogmaValidator):
        self.gene_manager = gene_manager
        self.expression_predictor = expression_predictor
        self.dogma_validator = dogma_validator
        
    def analyze_regulatory_landscape(self,
                                   elements: List[RegulatoryElement],
                                   cell_type: str = 'generic') -> Dict:
        """
        Complete integrated analysis of regulatory elements
        """
        results = {
            'gene_assignments': [],
            'expression_predictions': {},
            'validation_scores': {},
            'regulatory_networks': [],
            'summary_statistics': {}
        }
        
        # Step 1: Assign elements to genes
        interactions = self.gene_manager.assign_elements_to_genes(elements)
        results['gene_assignments'] = interactions
        
        # Step 2: Group by gene for analysis
        gene_elements = defaultdict(list)
        for interaction in interactions:
            gene_elements[interaction.gene].append(interaction)
            
        # Step 3: Analyze each gene
        for gene, gene_interactions in gene_elements.items():
            # Get elements for this gene
            gene_elements_list = [i.element for i in gene_interactions]
            
            # Add distance information to elements
            for interaction in gene_interactions:
                interaction.element.distance = interaction.distance
                interaction.element.interaction_type = interaction.interaction_type
                
            # Predict expression
            expression = self.expression_predictor.predict_expression_from_elements(
                gene, gene_elements_list, cell_type
            )
            
            results['expression_predictions'][gene.gene_id] = {
                'gene_name': gene.gene_name,
                'predicted_expression': expression,
                'num_regulatory_elements': len(gene_elements_list)
            }
            
            # Validate
            validation = self.dogma_validator.validate_gene_regulation(
                gene, gene_elements_list, expression
            )
            
            results['validation_scores'][gene.gene_id] = validation
            
        # Step 4: Build regulatory network
        results['regulatory_networks'] = self._build_regulatory_network(
            interactions
        )
        
        # Step 5: Summary statistics
        results['summary_statistics'] = self._compute_summary_stats(results)
        
        return results
        
    def _build_regulatory_network(self, 
                                interactions: List[RegulatoryInteraction]) -> List[Dict]:
        """Build regulatory network from interactions"""
        networks = []
        
        # Find TF genes
        tf_genes = set()
        all_genes = set()
        
        for interaction in interactions:
            gene = interaction.gene
            all_genes.add(gene.gene_name)
            
            if gene.gene_name in self.dogma_validator._get_tf_list():
                tf_genes.add(gene.gene_name)
                
        # Find regulatory connections
        for tf in tf_genes:
            tf_targets = []
            
            # Find elements with this TF's motifs
            for interaction in interactions:
                element = interaction.element
                gene = interaction.gene
                
                # Check if TF binds to this element
                for motif in element.tf_motifs:
                    if motif['tf'] == tf:
                        tf_targets.append({
                            'target_gene': gene.gene_name,
                            'element': f"{element.chr}:{element.start}-{element.end}",
                            'binding_score': motif['score']
                        })
                        
            if tf_targets:
                networks.append({
                    'tf': tf,
                    'targets': tf_targets
                })
                
        return networks
        
    def _compute_summary_stats(self, results: Dict) -> Dict:
        """Compute summary statistics"""
        stats = {}
        
        # Gene statistics
        stats['total_genes'] = len(results['expression_predictions'])
        stats['genes_with_enhancers'] = sum(
            1 for gene_data in results['gene_assignments']
            if any(i.element.element_type == 'enhancer' 
                  for i in results['gene_assignments']
                  if i.gene.gene_id == gene_data.gene.gene_id)
        )
        
        # Element statistics  
        all_elements = [i.element for i in results['gene_assignments']]
        unique_elements = list(set(e.start for e in all_elements))
        
        stats['total_elements'] = len(unique_elements)
        stats['elements_per_gene'] = len(all_elements) / max(stats['total_genes'], 1)
        
        # Expression statistics
        expressions = [v['predicted_expression'] 
                      for v in results['expression_predictions'].values()]
        if expressions:
            stats['mean_expression'] = np.mean(expressions)
            stats['expression_range'] = (min(expressions), max(expressions))
            
        # Validation statistics
        validation_scores = []
        for scores in results['validation_scores'].values():
            validation_scores.append(scores['regulatory_logic'])
            
        if validation_scores:
            stats['mean_validation_score'] = np.mean(validation_scores)
            
        # Network statistics
        stats['tf_genes'] = len(results['regulatory_networks'])
        stats['total_regulatory_edges'] = sum(
            len(net['targets']) for net in results['regulatory_networks']
        )
        
        return stats

# ======================== Evaluation Metrics ========================

class BiologicalEvaluator:
    """
    Evaluate predictions against known biological data
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_predictions(self, 
                           predictions: Dict,
                           ground_truth: Dict) -> Dict:
        """
        Comprehensive evaluation of predictions
        """
        evaluation = {}
        
        # Regulatory element detection metrics
        if 'elements' in ground_truth:
            element_metrics = self._evaluate_element_detection(
                predictions.get('elements', []),
                ground_truth['elements']
            )
            evaluation['element_detection'] = element_metrics
            
        # Gene assignment accuracy
        if 'gene_assignments' in ground_truth:
            assignment_metrics = self._evaluate_gene_assignments(
                predictions.get('gene_assignments', []),
                ground_truth['gene_assignments']
            )
            evaluation['gene_assignments'] = assignment_metrics
            
        # Expression prediction correlation
        if 'expression' in ground_truth:
            expression_metrics = self._evaluate_expression(
                predictions.get('expression_predictions', {}),
                ground_truth['expression']
            )
            evaluation['expression'] = expression_metrics
            
        return evaluation
        
    def _evaluate_element_detection(self, 
                                  predicted: List[RegulatoryElement],
                                  true_elements: List[Dict]) -> Dict:
        """Evaluate regulatory element detection"""
        # Convert to BED format for comparison
        pred_bed = pybedtools.BedTool(
            [(e.chr, e.start, e.end, e.element_type) for e in predicted]
        )
        
        true_bed = pybedtools.BedTool(
            [(e['chr'], e['start'], e['end'], e['type']) 
             for e in true_elements]
        )
        
        # Calculate overlaps
        overlaps = pred_bed.intersect(true_bed, wa=True, wb=True)
        
        # Compute metrics
        tp = len(overlaps)
        fp = len(pred_bed) - tp
        fn = len(true_bed) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
        
    def _evaluate_gene_assignments(self,
                                 predicted: List[RegulatoryInteraction],
                                 true_assignments: List[Dict]) -> Dict:
        """Evaluate gene assignment accuracy"""
        # Create prediction set
        pred_set = set()
        for interaction in predicted:
            key = (
                f"{interaction.element.chr}:{interaction.element.start}",
                interaction.gene.gene_id
            )
            pred_set.add(key)
            
        # Create ground truth set
        true_set = set()
        for assignment in true_assignments:
            key = (
                f"{assignment['element_chr']}:{assignment['element_start']}",
                assignment['gene_id']
            )
            true_set.add(key)
            
        # Calculate metrics
        correct = len(pred_set & true_set)
        predicted = len(pred_set)
        actual = len(true_set)
        
        precision = correct / predicted if predicted > 0 else 0
        recall = correct / actual if actual > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'correct_assignments': correct,
            'total_predicted': predicted,
            'total_actual': actual
        }
        
    def _evaluate_expression(self,
                           predicted: Dict[str, Dict],
                           true_expression: Dict[str, float]) -> Dict:
        """Evaluate expression predictions"""
        # Get matched predictions
        matched_genes = []
        pred_values = []
        true_values = []
        
        for gene_id, true_expr in true_expression.items():
            if gene_id in predicted:
                pred_expr = predicted[gene_id]['predicted_expression']
                pred_values.append(pred_expr)
                true_values.append(true_expr)
                matched_genes.append(gene_id)
                
        if not matched_genes:
            return {'error': 'No matched genes'}
            
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(pred_values, true_values)
        spearman_r, spearman_p = spearmanr(pred_values, true_values)
        
        # Mean squared error
        mse = np.mean((np.array(pred_values) - np.array(true_values))**2)
        
        return {
            'pearson_correlation': pearson_r,
            'pearson_pvalue': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_pvalue': spearman_p,
            'mse': mse,
            'matched_genes': len(matched_genes)
        }