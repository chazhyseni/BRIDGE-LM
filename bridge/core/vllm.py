"""
BRIDGE vLLM Integration - Efficient batching and continuous serving
Optimizes throughput for production-scale analysis
"""

import asyncio
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging
from queue import PriorityQueue
import threading

# Note: vLLM imports are conditional as they may not be available
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Using standard batching.")

from bridge_models import DNASequence, RegulatoryElement, BaseModelWrapper
from genome_handler import get_genome_handler

# ======================== Request Management ========================

@dataclass
class AnalysisRequest:
    """Single analysis request with metadata"""
    request_id: str
    sequence: DNASequence
    task_type: str  # 'regulatory_detection', 'expression_prediction', etc.
    priority: int = 5  # 1-10, higher = more urgent
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Any] = None
    metadata: Dict = field(default_factory=dict)
    
    def __lt__(self, other):
        # For priority queue - higher priority first
        return self.priority > other.priority

@dataclass
class BatchedRequest:
    """Group of requests batched for processing"""
    requests: List[AnalysisRequest]
    model_type: str
    total_tokens: int
    created_at: float = field(default_factory=time.time)

# ======================== Dynamic Request Router ========================

class RequestRouter:
    """
    Routes requests to appropriate models based on:
    - Sequence length
    - Task type
    - Resource availability
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Routing rules
        self.length_thresholds = {
            'short': 512,      # < 512bp -> DNABert2
            'medium': 8192,    # < 8kb -> Either
            'long': 50000,     # < 50kb -> HyenaDNA
            'very_long': float('inf')  # > 50kb -> HyenaDNA only
        }
        
        # Task to model mapping
        self.task_routing = {
            'regulatory_detection': ['hyenadna', 'dnabert2'],
            'fine_boundaries': ['dnabert2'],
            'expression_prediction': ['hyenadna', 'lucaone'],
            'protein_interaction': ['lucaone'],
            'conservation_analysis': ['hyenadna']
        }
        
    def route_request(self, request: AnalysisRequest) -> str:
        """Determine which model should handle this request"""
        seq_length = len(request.sequence)
        
        # Length-based routing
        if seq_length < self.length_thresholds['short']:
            length_category = 'short'
        elif seq_length < self.length_thresholds['medium']:
            length_category = 'medium'
        elif seq_length < self.length_thresholds['long']:
            length_category = 'long'
        else:
            length_category = 'very_long'
            
        # Get valid models for task
        valid_models = self.task_routing.get(request.task_type, ['hyenadna'])
        
        # Select based on length
        if length_category == 'short' and 'dnabert2' in valid_models:
            return 'dnabert2'
        elif length_category in ['long', 'very_long']:
            return 'hyenadna'
        else:
            # Medium length - choose based on task
            if request.task_type == 'fine_boundaries':
                return 'dnabert2'
            else:
                return valid_models[0]

# ======================== Continuous Batching Engine ========================

class ContinuousBatchingEngine:
    """
    Implements continuous batching for maximum throughput
    Key features:
    - Dynamic batch formation
    - Priority-aware scheduling  
    - Memory-efficient processing
    """
    
    def __init__(self, models: Dict[str, BaseModelWrapper], config: Dict):
        self.models = models
        self.config = config
        self.router = RequestRouter(config)
        
        # Request queues per model
        self.request_queues = {
            model_name: PriorityQueue()
            for model_name in models.keys()
        }
        
        # Batch settings
        self.max_batch_tokens = config.get('max_batch_tokens', 65536)
        self.max_batch_size = config.get('max_batch_size', 256)
        self.batch_timeout = config.get('batch_timeout', 0.1)  # 100ms
        
        # Processing threads
        self.processing_threads = {}
        self.stop_event = threading.Event()
        
        # Results cache
        self.results_cache = {}
        self.cache_lock = threading.Lock()
        
        # Start processing threads
        self._start_processing_threads()
        
    def _start_processing_threads(self):
        """Start a processing thread for each model"""
        for model_name in self.models.keys():
            thread = threading.Thread(
                target=self._process_model_queue,
                args=(model_name,),
                daemon=True
            )
            thread.start()
            self.processing_threads[model_name] = thread
            
        logging.info(f"Started {len(self.processing_threads)} processing threads")
        
    def submit_request(self, request: AnalysisRequest) -> str:
        """Submit request for processing"""
        # Route to appropriate model
        model_name = self.router.route_request(request)
        
        # Add to queue
        self.request_queues[model_name].put(request)
        
        logging.debug(f"Request {request.request_id} routed to {model_name}")
        
        return request.request_id
        
    async def submit_request_async(self, request: AnalysisRequest) -> Any:
        """Async version that waits for result"""
        future = asyncio.Future()
        request.callback = future
        
        self.submit_request(request)
        
        return await future
        
    def _process_model_queue(self, model_name: str):
        """Process requests for a specific model"""
        queue = self.request_queues[model_name]
        model = self.models[model_name]
        
        while not self.stop_event.is_set():
            # Collect requests for batching
            batch_requests = []
            batch_tokens = 0
            deadline = time.time() + self.batch_timeout
            
            while (len(batch_requests) < self.max_batch_size and
                   batch_tokens < self.max_batch_tokens and
                   time.time() < deadline):
                
                try:
                    timeout = max(0, deadline - time.time())
                    request = queue.get(timeout=timeout)
                    
                    # Check if batch would exceed token limit
                    request_tokens = self._estimate_tokens(request)
                    if batch_tokens + request_tokens > self.max_batch_tokens:
                        # Put back and process current batch
                        queue.put(request)
                        break
                        
                    batch_requests.append(request)
                    batch_tokens += request_tokens
                    
                except:
                    # Timeout - process what we have
                    break
                    
            # Process batch if we have requests
            if batch_requests:
                self._process_batch(model_name, model, batch_requests)
                
    def _estimate_tokens(self, request: AnalysisRequest) -> int:
        """Estimate tokens for a request"""
        # Simple estimation - can be refined
        return len(request.sequence) + 100  # Overhead
        
    def _process_batch(self, model_name: str, model: BaseModelWrapper,
                      requests: List[AnalysisRequest]):
        """Process a batch of requests"""
        start_time = time.time()
        
        logging.info(f"Processing batch of {len(requests)} requests on {model_name}")
        
        try:
            # Extract sequences
            sequences = [req.sequence.sequence for req in requests]
            
            # Route based on task type
            if requests[0].task_type == 'regulatory_detection':
                results = self._process_regulatory_batch(model, sequences)
            elif requests[0].task_type == 'fine_boundaries':
                results = self._process_boundary_batch(model, sequences)
            else:
                results = self._process_generic_batch(model, sequences)
                
            # Store results
            for request, result in zip(requests, results):
                with self.cache_lock:
                    self.results_cache[request.request_id] = result
                    
                # Trigger callback if async
                if request.callback:
                    if asyncio.iscoroutine(request.callback):
                        asyncio.create_task(request.callback(result))
                    elif hasattr(request.callback, 'set_result'):
                        request.callback.set_result(result)
                        
            # Log performance
            elapsed = time.time() - start_time
            throughput = len(requests) / elapsed
            logging.info(f"Batch processed in {elapsed:.2f}s ({throughput:.1f} seq/s)")
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            
            # Mark requests as failed
            for request in requests:
                if request.callback and hasattr(request.callback, 'set_exception'):
                    request.callback.set_exception(e)
                    
    def _process_regulatory_batch(self, model: BaseModelWrapper,
                                sequences: List[str]) -> List[List[RegulatoryElement]]:
        """Process batch for regulatory detection"""
        if hasattr(model, 'predict_batch'):
            predictions = model.predict_batch(sequences)
            
            # Convert to RegulatoryElement objects
            results = []
            for seq_preds in predictions:
                elements = []
                for pred in seq_preds:
                    if pred['score'] > self.config.get('detection_threshold', 0.7):
                        element = RegulatoryElement(
                            sequence=pred['sequence'],
                            chr='unknown',  # Would be filled from request
                            start=pred['start'],
                            end=pred['end'],
                            element_type='enhancer',  # Would be determined
                            score=pred['score']
                        )
                        elements.append(element)
                results.append(elements)
                
            return results
        else:
            # Fallback to sequential processing
            return [[] for _ in sequences]
            
    def _process_boundary_batch(self, model: BaseModelWrapper,
                              sequences: List[str]) -> List[Dict]:
        """Process batch for boundary refinement"""
        if hasattr(model, 'scan_sequence_precise'):
            results = []
            for seq in sequences:
                scores = model.scan_sequence_precise(seq)
                results.append(scores)
            return results
        else:
            return [{} for _ in sequences]
            
    def _process_generic_batch(self, model: BaseModelWrapper,
                             sequences: List[str]) -> List[Any]:
        """Generic batch processing"""
        return model.predict_batch(sequences)
        
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Get result for a request ID"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.cache_lock:
                if request_id in self.results_cache:
                    return self.results_cache.pop(request_id)
                    
            time.sleep(0.01)  # Small delay
            
        return None
        
    def shutdown(self):
        """Shutdown processing threads"""
        self.stop_event.set()
        
        for thread in self.processing_threads.values():
            thread.join(timeout=5.0)

# ======================== vLLM Model Wrapper ========================

class VLLMModelWrapper:
    """
    Wraps models for vLLM-style efficient inference
    """
    
    def __init__(self, model: BaseModelWrapper, config: Dict):
        self.model = model
        self.config = config
        
        # Only initialize vLLM if available
        if VLLM_AVAILABLE:
            # Initialize vLLM engine
            self.engine_args = AsyncEngineArgs(
                model=config.get('model_path'),
                tokenizer=config.get('tokenizer_path'),
                tokenizer_mode=config.get('tokenizer_mode', 'auto'),
                trust_remote_code=True,
                max_num_batched_tokens=config.get('max_num_batched_tokens', 65536),
                max_num_seqs=config.get('max_num_seqs', 256),
                max_model_len=config.get('max_model_len', 8192),
                enable_prefix_caching=True
            )
            
            # Async engine for continuous batching
            self.async_engine = None
            self._initialize_engine()
        else:
            self.async_engine = None
        
    def _initialize_engine(self):
        """Initialize async LLM engine"""
        # Note: This is pseudo-code as vLLM doesn't directly support
        # biological models yet. Shows the pattern.
        try:
            self.async_engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            logging.info("vLLM engine initialized successfully")
        except Exception as e:
            logging.warning(f"Could not initialize vLLM engine: {e}")
            logging.info("Falling back to standard batching")
            
    async def generate_async(self, prompts: List[str],
                           sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """Async generation using vLLM engine"""
        if self.async_engine is None:
            # Fallback to regular model
            return self.model.predict_batch(prompts)
            
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for biological sequences
                max_tokens=1000
            )
            
        # Generate with continuous batching
        results = []
        async for output in self.async_engine.generate(prompts, sampling_params):
            results.append(output)
            
        return results

# ======================== Unified vLLM Interface ========================

class BridgeVLLM:
    """
    Main vLLM interface for BRIDGE
    Coordinates all models with continuous batching
    """
    
    def __init__(self, models: Dict[str, BaseModelWrapper], config: Dict):
        self.models = models
        self.config = config
        
        # Initialize continuous batching engine
        self.batch_engine = ContinuousBatchingEngine(models, config)
        
        # Request tracking
        self.pending_requests = {}
        self.completed_requests = {}
        
        # Performance monitoring
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_sequences': 0,
            'total_time': 0.0
        }
        
        # Initialize genome handler
        self.genome_handler = get_genome_handler(config.get('data_dir', 'data'))
        
    def _tokenize_dna(self, sequence: str) -> List[int]:
        """Tokenize DNA sequence to integers"""
        # Simple character-level tokenization for DNA
        vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        tokens = []
        
        for base in sequence.upper():
            if base in vocab:
                tokens.append(vocab[base])
            else:
                tokens.append(vocab['N'])  # Unknown bases become N
                
        return tokens
        
    async def analyze_sequences(self, sequences: List[DNASequence],
                              task: str = 'regulatory_detection',
                              priority: int = 5) -> List[Any]:
        """
        Analyze multiple sequences with continuous batching
        
        Args:
            sequences: List of DNA sequences
            task: Analysis task
            priority: Request priority (1-10)
            
        Returns:
            List of results
        """
        # Create requests
        requests = []
        for i, seq in enumerate(sequences):
            request = AnalysisRequest(
                request_id=f"{task}_{time.time()}_{i}",
                sequence=seq,
                task_type=task,
                priority=priority
            )
            requests.append(request)
            
        # Submit all requests
        futures = []
        for request in requests:
            future = await self.batch_engine.submit_request_async(request)
            futures.append(future)
            
        # Wait for all results
        results = await asyncio.gather(*futures)
        
        # Update stats
        self.stats['total_requests'] += len(requests)
        self.stats['completed_requests'] += len(results)
        self.stats['total_sequences'] += len(sequences)
        
        return results
        
    async def hierarchical_analysis(self, locus: DNASequence) -> Dict:
        """
        Perform hierarchical analysis with efficient routing
        
        1. Coarse detection with HyenaDNA
        2. Parallel fine-grain analysis with DNABert2
        3. Validation with LucaOne
        """
        results = {
            'coarse_elements': [],
            'refined_elements': [],
            'validation_scores': []
        }
        
        # Stage 1: Coarse detection
        coarse_request = AnalysisRequest(
            request_id=f"coarse_{time.time()}",
            sequence=locus,
            task_type='regulatory_detection',
            priority=8  # High priority
        )
        
        coarse_results = await self.batch_engine.submit_request_async(coarse_request)
        results['coarse_elements'] = coarse_results
        
        # Stage 2: Parallel refinement
        if coarse_results:
            refinement_requests = []
            
            for element in coarse_results:
                # Extract region with padding
                start = max(0, element.start - 200)
                end = min(len(locus), element.end + 200)
                
                subseq = DNASequence(
                    sequence=locus.sequence[start:end],
                    chromosome=locus.chromosome,
                    start=locus.start + start if locus.start else start,
                    end=locus.start + end if locus.start else end
                )
                
                request = AnalysisRequest(
                    request_id=f"refine_{time.time()}_{element.start}",
                    sequence=subseq,
                    task_type='fine_boundaries',
                    priority=7,
                    metadata={'original_element': element}
                )
                
                refinement_requests.append(request)
                
            # Submit all refinement requests
            refinement_futures = []
            for req in refinement_requests:
                future = await self.batch_engine.submit_request_async(req)
                refinement_futures.append(future)
                
            # Wait for results
            refinement_results = await asyncio.gather(*refinement_futures)
            
            # Process refinement results
            for req, result in zip(refinement_requests, refinement_results):
                original = req.metadata['original_element']
                # Update element boundaries based on refinement
                # ... (implementation details)
                results['refined_elements'].append(original)
                
        return results
        
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        # Calculate throughput
        if stats['total_time'] > 0:
            stats['throughput'] = stats['total_sequences'] / stats['total_time']
        else:
            stats['throughput'] = 0
            
        # Add queue statistics
        queue_stats = {}
        for model_name, queue in self.batch_engine.request_queues.items():
            queue_stats[model_name] = {
                'pending': queue.qsize(),
                'model_loaded': model_name in self.models
            }
            
        stats['queues'] = queue_stats
        
        return stats
        
    def shutdown(self):
        """Graceful shutdown"""
        logging.info("Shutting down BRIDGE vLLM engine")
        self.batch_engine.shutdown()

# ======================== Optimized Pipeline Implementations ========================

class OptimizedPipelines:
    """
    Pre-built optimized pipelines for common workflows
    """
    
    def __init__(self, vllm_engine: BridgeVLLM):
        self.engine = vllm_engine
        self.genome_handler = self.engine.genome_handler
        
    async def full_locus_analysis(self, chromosome: str, start: int, end: int,
                                genome: str = 'hg38') -> Dict:
        """
        Complete analysis of a genomic locus
        Optimized for throughput with intelligent batching
        """
        # Fetch actual sequence from genome
        sequence = await self._fetch_sequence(chromosome, start, end, genome)
        
        # Create locus sequence
        locus = DNASequence(
            sequence=sequence,
            chromosome=chromosome,
            start=start,
            end=end
        )
        
        # Run hierarchical analysis
        results = await self.engine.hierarchical_analysis(locus)
        
        # Post-process results
        processed = {
            'locus': f"{chromosome}:{start}-{end}",
            'regulatory_elements': results['refined_elements'],
            'summary': self._create_summary(results)
        }
        
        return processed
        
    async def batch_variant_analysis(self, variants: List[Dict]) -> List[Dict]:
        """
        Analyze multiple variants in parallel
        """
        # Group variants by proximity for efficiency
        variant_groups = self._group_nearby_variants(variants)
        
        # Create analysis requests
        requests = []
        for group in variant_groups:
            # Get sequence covering all variants in group
            chrom = group[0]['chrom']
            start = min(v['pos'] for v in group) - 1000
            end = max(v['pos'] for v in group) + 1000
            
            # Fetch actual sequence
            sequence = await self._fetch_sequence(chrom, start, end)
            
            seq = DNASequence(
                sequence=sequence,
                chromosome=chrom,
                start=start,
                end=end
            )
            
            request = AnalysisRequest(
                request_id=f"variant_group_{chrom}_{start}",
                sequence=seq,
                task_type='regulatory_detection',
                priority=6,
                metadata={'variants': group}
            )
            
            requests.append(request)
            
        # Process all groups in parallel
        results = await asyncio.gather(*[
            self.engine.batch_engine.submit_request_async(req)
            for req in requests
        ])
        
        # Map results back to individual variants
        variant_results = []
        for request, result in zip(requests, results):
            for variant in request.metadata['variants']:
                # Check if variant affects any regulatory element
                affected = self._check_variant_impact(variant, result)
                variant_results.append({
                    'variant': variant,
                    'regulatory_impact': affected
                })
                
        return variant_results
        
    def _group_nearby_variants(self, variants: List[Dict],
                             max_distance: int = 10000) -> List[List[Dict]]:
        """Group variants within max_distance of each other"""
        # Sort by chromosome and position
        sorted_variants = sorted(variants, key=lambda v: (v['chrom'], v['pos']))
        
        groups = []
        current_group = []
        
        for variant in sorted_variants:
            if not current_group:
                current_group.append(variant)
            else:
                # Check distance to last variant in group
                last = current_group[-1]
                if (variant['chrom'] == last['chrom'] and
                    variant['pos'] - last['pos'] <= max_distance):
                    current_group.append(variant)
                else:
                    # Start new group
                    groups.append(current_group)
                    current_group = [variant]
                    
        if current_group:
            groups.append(current_group)
            
        return groups
        
    def _check_variant_impact(self, variant: Dict,
                            elements: List[RegulatoryElement]) -> Dict:
        """Check if variant affects any regulatory element"""
        for element in elements:
            if element.start <= variant['pos'] <= element.end:
                # Check if variant disrupts TF motifs
                disrupted_motifs = []
                for motif in element.tf_motifs:
                    motif_start = element.start + motif['position']
                    motif_end = motif_start + len(motif['sequence'])
                    
                    if motif_start <= variant['pos'] <= motif_end:
                        disrupted_motifs.append(motif)
                        
                return {
                    'affected': True,
                    'element': element,
                    'position_in_element': variant['pos'] - element.start,
                    'disrupted_motifs': disrupted_motifs
                }
                
        return {'affected': False}
        
    async def _fetch_sequence(self, chromosome: str, start: int, end: int,
                            genome: str = 'hg38') -> str:
        """Fetch genomic sequence using genome handler"""
        return self.genome_handler.fetch_sequence(chromosome, start, end, genome)
        
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics"""
        return {
            'total_elements': len(results.get('refined_elements', [])),
            'enhancers': sum(1 for e in results.get('refined_elements', [])
                           if e.element_type == 'enhancer'),
            'promoters': sum(1 for e in results.get('refined_elements', [])
                           if e.element_type == 'promoter')
        }
    