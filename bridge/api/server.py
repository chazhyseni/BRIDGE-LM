"""
BRIDGE API Server - Production-ready REST API and main workflow orchestration
Provides endpoints for regulatory analysis, batch processing, and result retrieval
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import uvicorn
from datetime import datetime
import uuid
import os
import json
import logging
from pathlib import Path
import tempfile
import shutil

# Import BRIDGE components
from bridge_models import (
    DNASequence, RegulatoryElement, BridgeModelManager
)
from bridge_refiner import BoundaryRefiner
from bridge_vllm import BridgeVLLM, OptimizedPipelines
from bridge_bio import (
    GeneAnnotationManager, ExpressionPredictor,
    CentralDogmaValidator, IntegratedBiologicalAnalysis
)

# ======================== Request/Response Models ========================

class SequenceRequest(BaseModel):
    """Request for single sequence analysis"""
    sequence: str = Field(..., min_length=100, max_length=1000000)
    chromosome: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    analysis_type: str = Field('full', regex='^(full|regulatory|expression|interaction)$')
    parameters: Optional[Dict[str, Any]] = None

class BatchRequest(BaseModel):
    """Request for batch analysis"""
    sequences: List[SequenceRequest]
    priority: int = Field(5, ge=1, le=10)
    callback_url: Optional[str] = None

class GenomicRegionRequest(BaseModel):
    """Request for genomic region analysis"""
    genome: str = Field('hg38', regex='^(hg38|hg19|mm10)$')
    chromosome: str
    start: int = Field(..., gt=0)
    end: int = Field(..., gt=0)
    include_genes: bool = True
    include_expression: bool = True
    cell_type: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Response for analysis request"""
    request_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ======================== BRIDGE Application ========================

class BridgeApplication:
    """
    Main BRIDGE application coordinating all components
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.initialized = False
        
        # Component storage
        self.model_manager = None
        self.boundary_refiner = None
        self.vllm_engine = None
        self.gene_manager = None
        self.expression_predictor = None
        self.dogma_validator = None
        self.bio_analyzer = None
        self.pipelines = None
        
        # Job tracking
        self.jobs = {}
        self.results_dir = Path(self.config.get('results_dir', './results'))
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
            
    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
            
        logging.info("Initializing BRIDGE components...")
        
        # Initialize models
        self.model_manager = BridgeModelManager(self.config['models'])
        
        # Initialize refinement engine
        self.boundary_refiner = BoundaryRefiner(self.model_manager.models)
        
        # Initialize vLLM engine
        self.vllm_engine = BridgeVLLM(
            self.model_manager.models,
            self.config['vllm']
        )
        
        # Initialize biological components
        self.gene_manager = GeneAnnotationManager(
            self.config['gene_annotation_file'],
            genome=self.config.get('genome', 'hg38')
        )
        
        self.expression_predictor = ExpressionPredictor(self.model_manager)
        
        if 'lucaone' in self.model_manager.models:
            self.dogma_validator = CentralDogmaValidator(
                self.model_manager.models['lucaone']
            )
        
        self.bio_analyzer = IntegratedBiologicalAnalysis(
            self.gene_manager,
            self.expression_predictor,
            self.dogma_validator
        )
        
        # Initialize optimized pipelines
        self.pipelines = OptimizedPipelines(self.vllm_engine)
        
        self.initialized = True
        logging.info("BRIDGE initialization complete")
        
    async def analyze_sequence(self, request: SequenceRequest) -> Dict:
        """
        Analyze a single sequence
        """
        # Create DNASequence object
        dna_seq = DNASequence(
            sequence=request.sequence,
            chromosome=request.chromosome,
            start=request.start,
            end=request.end
        )
        
        # Run analysis based on type
        if request.analysis_type == 'full':
            return await self._full_analysis(dna_seq, request.parameters)
        elif request.analysis_type == 'regulatory':
            return await self._regulatory_analysis(dna_seq, request.parameters)
        elif request.analysis_type == 'expression':
            return await self._expression_analysis(dna_seq, request.parameters)
        elif request.analysis_type == 'interaction':
            return await self._interaction_analysis(dna_seq, request.parameters)
            
    async def _full_analysis(self, sequence: DNASequence, 
                           parameters: Optional[Dict] = None) -> Dict:
        """Complete analysis pipeline"""
        results = {}
        
        # Step 1: Detect regulatory elements
        elements = await self.vllm_engine.analyze_sequences(
            [sequence],
            task='regulatory_detection'
        )
        
        if elements and elements[0]:
            results['regulatory_elements'] = [
                self._element_to_dict(e) for e in elements[0]
            ]
            
            # Step 2: Refine boundaries
            refined = self.boundary_refiner.batch_refine_elements(
                elements[0],
                [sequence] * len(elements[0])
            )
            
            results['refined_elements'] = [
                self._element_to_dict(e) for e in refined
            ]
            
            # Step 3: Biological analysis
            bio_results = self.bio_analyzer.analyze_regulatory_landscape(
                refined,
                cell_type=parameters.get('cell_type', 'generic') if parameters else 'generic'
            )
            
            results['biological_analysis'] = bio_results
            
        return results
        
    async def _regulatory_analysis(self, sequence: DNASequence,
                                 parameters: Optional[Dict] = None) -> Dict:
        """Regulatory element detection only"""
        # Detect elements
        elements = await self.vllm_engine.analyze_sequences(
            [sequence],
            task='regulatory_detection'
        )
        
        results = {'elements': []}
        
        if elements and elements[0]:
            # Refine if requested
            if parameters and parameters.get('refine_boundaries', True):
                refined = self.boundary_refiner.batch_refine_elements(
                    elements[0],
                    [sequence] * len(elements[0])
                )
                results['elements'] = [self._element_to_dict(e) for e in refined]
            else:
                results['elements'] = [self._element_to_dict(e) for e in elements[0]]
                
        return results
        
    async def _expression_analysis(self, sequence: DNASequence,
                                 parameters: Optional[Dict] = None) -> Dict:
        """Expression prediction analysis"""
        # First detect regulatory elements
        elements = await self.vllm_engine.analyze_sequences(
            [sequence],
            task='regulatory_detection'
        )
        
        if not elements or not elements[0]:
            return {'error': 'No regulatory elements found'}
            
        # Find genes in region
        genes_in_region = self.gene_manager.find_genes_in_region(
            sequence.chromosome,
            sequence.start,
            sequence.end
        )
        
        if not genes_in_region:
            return {'error': 'No genes found in region'}
            
        # Predict expression
        expression_results = {}
        
        for gene in genes_in_region:
            expression = self.expression_predictor.predict_expression_from_elements(
                gene,
                elements[0],
                cell_type=parameters.get('cell_type', 'generic') if parameters else 'generic'
            )
            
            expression_results[gene.gene_id] = {
                'gene_name': gene.gene_name,
                'predicted_expression': expression,
                'regulatory_elements': len(elements[0])
            }
            
        return {'expression_predictions': expression_results}
        
    async def _interaction_analysis(self, sequence: DNASequence,
                                  parameters: Optional[Dict] = None) -> Dict:
        """DNA-protein interaction analysis"""
        if not parameters or 'protein_sequence' not in parameters:
            return {'error': 'Protein sequence required for interaction analysis'}
            
        # Use LucaOne for interaction prediction
        if 'lucaone' not in self.model_manager.models:
            return {'error': 'LucaOne model required for interaction analysis'}
            
        lucaone = self.model_manager.models['lucaone']
        
        # Predict interaction
        interaction_score = lucaone._verify_central_dogma(
            sequence.sequence,
            parameters['protein_sequence']
        )
        
        # Find potential binding sites
        binding_sites = []
        window_size = 20
        stride = 5
        
        for i in range(0, len(sequence.sequence) - window_size, stride):
            window = sequence.sequence[i:i + window_size]
            
            score = lucaone._verify_central_dogma(
                window,
                parameters['protein_sequence']
            )
            
            if score > 0.7:
                binding_sites.append({
                    'start': sequence.start + i if sequence.start else i,
                    'end': sequence.start + i + window_size if sequence.start else i + window_size,
                    'score': score,
                    'sequence': window
                })
                
        return {
            'overall_interaction_score': interaction_score,
            'binding_sites': binding_sites
        }
        
    def _element_to_dict(self, element: RegulatoryElement) -> Dict:
        """Convert RegulatoryElement to dictionary"""
        return {
            'chromosome': element.chr,
            'start': element.start,
            'end': element.end,
            'type': element.element_type,
            'score': element.score,
            'length': element.length,
            'core_start': element.core_start,
            'core_end': element.core_end,
            'tf_motifs': element.tf_motifs,
            'conservation_score': element.conservation_score,
            'sequence': element.sequence[:50] + '...' if len(element.sequence) > 50 else element.sequence
        }
        
    async def process_batch(self, batch_request: BatchRequest) -> str:
        """Process batch of sequences"""
        job_id = str(uuid.uuid4())
        
        # Create job entry
        self.jobs[job_id] = {
            'status': 'pending',
            'created_at': datetime.now(),
            'total_sequences': len(batch_request.sequences),
            'completed_sequences': 0,
            'results': []
        }
        
        # Process in background
        asyncio.create_task(self._process_batch_async(job_id, batch_request))
        
        return job_id
        
    async def _process_batch_async(self, job_id: str, batch_request: BatchRequest):
        """Async batch processing"""
        try:
            self.jobs[job_id]['status'] = 'processing'
            results = []
            
            for i, seq_request in enumerate(batch_request.sequences):
                try:
                    result = await self.analyze_sequence(seq_request)
                    results.append({
                        'sequence_index': i,
                        'status': 'success',
                        'results': result
                    })
                except Exception as e:
                    results.append({
                        'sequence_index': i,
                        'status': 'error',
                        'error': str(e)
                    })
                    
                self.jobs[job_id]['completed_sequences'] += 1
                
            # Save results
            self.jobs[job_id]['results'] = results
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['completed_at'] = datetime.now()
            
            # Save to file
            result_file = self.results_dir / f"{job_id}.json"
            with open(result_file, 'w') as f:
                json.dump(self.jobs[job_id], f, indent=2, default=str)
                
            # Callback if provided
            if batch_request.callback_url:
                await self._send_callback(batch_request.callback_url, job_id)
                
        except Exception as e:
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
            
    async def _send_callback(self, callback_url: str, job_id: str):
        """Send callback notification"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    callback_url,
                    json={
                        'job_id': job_id,
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logging.error(f"Callback failed: {str(e)}")

# ======================== FastAPI Application ========================

# Create app instance
app = FastAPI(
    title="BRIDGE API",
    description="Biological Regulatory Integration & Detection for Gene Expression",
    version="1.0.0"
)

# Create BRIDGE instance
bridge = BridgeApplication("config/bridge_config.json")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize BRIDGE on startup"""
    await bridge.initialize()

# ======================== API Endpoints ========================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "BRIDGE API",
        "version": "1.0.0",
        "status": "running" if bridge.initialized else "initializing"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sequence(request: SequenceRequest):
    """
    Analyze a single DNA sequence
    
    Analysis types:
    - full: Complete regulatory + expression analysis
    - regulatory: Only regulatory element detection
    - expression: Expression prediction (requires gene context)
    - interaction: DNA-protein interaction analysis
    """
    if not bridge.initialized:
        raise HTTPException(status_code=503, detail="Service initializing")
        
    try:
        # Create analysis ID
        request_id = str(uuid.uuid4())
        
        # Run analysis
        results = await bridge.analyze_sequence(request)
        
        # Create response
        response = AnalysisResponse(
            request_id=request_id,
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            results=results
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def submit_batch(batch_request: BatchRequest):
    """
    Submit batch analysis job
    
    Returns job ID for tracking
    """
    if not bridge.initialized:
        raise HTTPException(status_code=503, detail="Service initializing")
        
    job_id = await bridge.process_batch(batch_request)
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "total_sequences": len(batch_request.sequences)
    }

@app.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get batch job status"""
    if job_id not in bridge.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = bridge.jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job['status'],
        "created_at": job['created_at'],
        "total_sequences": job['total_sequences'],
        "completed_sequences": job['completed_sequences'],
        "completed_at": job.get('completed_at')
    }

@app.get("/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    """Get batch job results"""
    if job_id not in bridge.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = bridge.jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
        
    return job['results']

@app.post("/region", response_model=AnalysisResponse)
async def analyze_genomic_region(request: GenomicRegionRequest):
"""
    Analyze a genomic region
    
    Fetches sequence from genome and performs complete analysis
    """
    if not bridge.initialized:
        raise HTTPException(status_code=503, detail="Service initializing")
        
    try:
        # Use optimized pipeline for region analysis
        results = await bridge.pipelines.full_locus_analysis(
            chromosome=request.chromosome,
            start=request.start,
            end=request.end,
            genome=request.genome
        )
        
        # Add gene information if requested
        if request.include_genes:
            genes = bridge.gene_manager.find_genes_in_region(
                request.chromosome,
                request.start,
                request.end
            )
            
            results['genes'] = [
                {
                    'gene_id': g.gene_id,
                    'gene_name': g.gene_name,
                    'type': g.gene_type,
                    'start': g.start,
                    'end': g.end,
                    'strand': g.strand
                }
                for g in genes
            ]
            
        # Add expression predictions if requested
        if request.include_expression and 'regulatory_elements' in results:
            expression_results = bridge.bio_analyzer.analyze_regulatory_landscape(
                results['regulatory_elements'],
                cell_type=request.cell_type or 'generic'
            )
            
            results['expression_analysis'] = expression_results
            
        # Create response
        response = AnalysisResponse(
            request_id=str(uuid.uuid4()),
            status="completed",
            created_at=datetime.now(),
            completed_at=datetime.now(),
            results=results
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Region analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/variants")
async def analyze_variants(variants: List[Dict]):
    """
    Analyze regulatory impact of genetic variants
    
    Input format:
    [
        {
            "chrom": "chr1",
            "pos": 12345,
            "ref": "A",
            "alt": "G"
        }
    ]
    """
    if not bridge.initialized:
        raise HTTPException(status_code=503, detail="Service initializing")
        
    try:
        # Use batch variant analysis
        results = await bridge.pipelines.batch_variant_analysis(variants)
        
        return {
            "total_variants": len(variants),
            "variants_affecting_regulatory": sum(
                1 for r in results if r['regulatory_impact']['affected']
            ),
            "detailed_results": results
        }
        
    except Exception as e:
        logging.error(f"Variant analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/fasta")
async def upload_fasta(file: UploadFile = File(...)):
    """
    Upload FASTA file for analysis
    
    Returns job ID for tracking
    """
    if not bridge.initialized:
        raise HTTPException(status_code=503, detail="Service initializing")
        
    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        # Save file
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
            
        # Parse FASTA
        from Bio import SeqIO
        sequences = []
        
        for record in SeqIO.parse(temp_path, "fasta"):
            sequences.append(SequenceRequest(
                sequence=str(record.seq),
                chromosome=record.id,
                analysis_type='full'
            ))
            
        # Create batch request
        batch_request = BatchRequest(sequences=sequences)
        
        # Submit for processing
        job_id = await bridge.process_batch(batch_request)
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "sequences_found": len(sequences)
        }
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

@app.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    if not bridge.initialized:
        return {"status": "initializing"}
        
    model_info = {}
    
    for name, model in bridge.model_manager.models.items():
        model_info[name] = {
            'loaded': True,
            'device': str(model.device),
            'max_length': getattr(model, 'max_length', 'N/A')
        }
        
    return {
        'models': model_info,
        'vllm_enabled': bridge.vllm_engine is not None,
        'gene_annotations_loaded': bridge.gene_manager is not None
    }

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    stats = bridge.vllm_engine.get_statistics() if bridge.vllm_engine else {}
    
    # Add job statistics
    stats['jobs'] = {
        'total': len(bridge.jobs),
        'pending': sum(1 for j in bridge.jobs.values() if j['status'] == 'pending'),
        'processing': sum(1 for j in bridge.jobs.values() if j['status'] == 'processing'),
        'completed': sum(1 for j in bridge.jobs.values() if j['status'] == 'completed'),
        'failed': sum(1 for j in bridge.jobs.values() if j['status'] == 'failed')
    }
    
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if bridge.initialized else "initializing",
        "timestamp": datetime.now().isoformat()
    }

# ======================== Utility Endpoints ========================

@app.post("/utils/sequence_stats")
async def get_sequence_statistics(sequence: str):
    """Get basic statistics about a DNA sequence"""
    # GC content
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0
    
    # N content
    n_count = sequence.count('N')
    n_content = n_count / len(sequence) if len(sequence) > 0 else 0
    
    # Find repeats
    repeats = []
    for k in [2, 3, 4]:  # Check 2-4 mers
        kmer_counts = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            
        # Report high-frequency kmers
        for kmer, count in kmer_counts.items():
            if count > len(sequence) / (4**k) * 5:  # 5x expected
                repeats.append({
                    'sequence': kmer,
                    'count': count,
                    'frequency': count / (len(sequence) - k + 1)
                })
                
    return {
        'length': len(sequence),
        'gc_content': gc_content,
        'n_content': n_content,
        'repeats': sorted(repeats, key=lambda x: x['count'], reverse=True)[:10]
    }

@app.get("/download/{job_id}")
async def download_results(job_id: str, format: str = "json"):
    """Download analysis results"""
    if job_id not in bridge.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = bridge.jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
        
    # Generate file based on format
    if format == "json":
        result_file = bridge.results_dir / f"{job_id}.json"
        if not result_file.exists():
            with open(result_file, 'w') as f:
                json.dump(job, f, indent=2, default=str)
                
        return FileResponse(
            result_file,
            media_type="application/json",
            filename=f"bridge_results_{job_id}.json"
        )
        
    elif format == "bed":
        # Convert to BED format
        bed_file = bridge.results_dir / f"{job_id}.bed"
        
        with open(bed_file, 'w') as f:
            f.write(f"track name=BRIDGE_regulatory_elements_{job_id}\n")
            
            for seq_result in job.get('results', []):
                if seq_result['status'] == 'success':
                    results = seq_result.get('results', {})
                    
                    # Extract elements
                    elements = results.get('refined_elements', results.get('regulatory_elements', []))
                    
                    for elem in elements:
                        f.write(f"{elem['chromosome']}\t{elem['start']}\t{elem['end']}\t"
                               f"{elem['type']}\t{int(elem['score']*1000)}\t.\n")
                               
        return FileResponse(
            bed_file,
            media_type="text/plain",
            filename=f"bridge_elements_{job_id}.bed"
        )
        
    else:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'bed'")

# ======================== Main Workflow Functions ========================

async def complete_analysis_workflow(
    genome_file: str,
    output_dir: str,
    chromosomes: Optional[List[str]] = None,
    chunk_size: int = 1000000,
    overlap: int = 10000
):
    """
    Complete genome-wide analysis workflow
    
    Args:
        genome_file: Path to genome FASTA
        output_dir: Output directory
        chromosomes: List of chromosomes to analyze (None = all)
        chunk_size: Size of chunks to process
        overlap: Overlap between chunks
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize components
    await bridge.initialize()
    
    # Parse genome
    from Bio import SeqIO
    
    # Track all jobs
    all_jobs = []
    
    for record in SeqIO.parse(genome_file, "fasta"):
        chrom = record.id
        
        # Skip if not in selected chromosomes
        if chromosomes and chrom not in chromosomes:
            continue
            
        logging.info(f"Processing chromosome {chrom}")
        
        # Process in chunks
        seq_len = len(record.seq)
        
        for start in range(0, seq_len, chunk_size - overlap):
            end = min(start + chunk_size, seq_len)
            
            # Create request
            seq_request = SequenceRequest(
                sequence=str(record.seq[start:end]),
                chromosome=chrom,
                start=start,
                end=end,
                analysis_type='full'
            )
            
            # Submit for analysis
            batch_request = BatchRequest(
                sequences=[seq_request],
                priority=7
            )
            
            job_id = await bridge.process_batch(batch_request)
            all_jobs.append({
                'job_id': job_id,
                'chromosome': chrom,
                'start': start,
                'end': end
            })
            
    # Wait for all jobs to complete
    logging.info(f"Submitted {len(all_jobs)} jobs, waiting for completion...")
    
    completed = 0
    while completed < len(all_jobs):
        await asyncio.sleep(5)
        
        completed = sum(
            1 for job in all_jobs
            if bridge.jobs.get(job['job_id'], {}).get('status') == 'completed'
        )
        
        logging.info(f"Progress: {completed}/{len(all_jobs)} jobs completed")
        
    # Merge results
    logging.info("Merging results...")
    
    all_elements = []
    all_genes = {}
    all_expression = {}
    
    for job_info in all_jobs:
        job = bridge.jobs[job_info['job_id']]
        
        if job['status'] == 'completed' and job.get('results'):
            for seq_result in job['results']:
                if seq_result['status'] == 'success':
                    results = seq_result['results']
                    
                    # Extract elements
                    if 'refined_elements' in results:
                        all_elements.extend(results['refined_elements'])
                        
                    # Extract gene assignments
                    if 'biological_analysis' in results:
                        bio = results['biological_analysis']
                        
                        if 'expression_predictions' in bio:
                            all_expression.update(bio['expression_predictions'])
                            
    # Write merged results
    output_file = output_path / "regulatory_elements.bed"
    with open(output_file, 'w') as f:
        f.write("track name=BRIDGE_regulatory_elements\n")
        f.write("#chrom\tstart\tend\ttype\tscore\tstrand\n")
        
        for elem in sorted(all_elements, key=lambda x: (x['chromosome'], x['start'])):
            f.write(f"{elem['chromosome']}\t{elem['start']}\t{elem['end']}\t"
                   f"{elem['type']}\t{int(elem['score']*1000)}\t.\n")
                   
    # Write expression predictions
    expr_file = output_path / "expression_predictions.tsv"
    with open(expr_file, 'w') as f:
        f.write("gene_id\tgene_name\tpredicted_expression\tnum_regulatory_elements\n")
        
        for gene_id, expr_data in all_expression.items():
            f.write(f"{gene_id}\t{expr_data['gene_name']}\t"
                   f"{expr_data['predicted_expression']:.4f}\t"
                   f"{expr_data['num_regulatory_elements']}\n")
                   
    # Summary statistics
    summary = {
        'total_elements': len(all_elements),
        'enhancers': sum(1 for e in all_elements if e['type'] == 'enhancer'),
        'promoters': sum(1 for e in all_elements if e['type'] == 'promoter'),
        'genes_analyzed': len(all_expression),
        'mean_elements_per_gene': len(all_elements) / max(len(all_expression), 1)
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    logging.info(f"Analysis complete. Results saved to {output_dir}")
    
    return summary

# ======================== CLI Interface ========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BRIDGE API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="config/bridge_config.json", help="Config file")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Add workflow mode
    parser.add_argument("--workflow", action="store_true", help="Run workflow mode")
    parser.add_argument("--genome", help="Genome FASTA file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--chromosomes", nargs="+", help="Chromosomes to process")
    
    args = parser.parse_args()
    
    if args.workflow:
        # Run workflow mode
        if not args.genome or not args.output:
            parser.error("--genome and --output required for workflow mode")
            
        # Run workflow
        asyncio.run(complete_analysis_workflow(
            args.genome,
            args.output,
            args.chromosomes
        ))
    else:
        # Run API server
        uvicorn.run(
            "bridge_api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level="info"
        )