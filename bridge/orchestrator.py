#!/usr/bin/env python3
"""
BRIDGE Orchestrator - Central workflow management system
Coordinates all BRIDGE components for seamless analysis workflows
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import defaultdict

# Import BRIDGE components
from bridge_models import BridgeModelManager, DNASequence, RegulatoryElement
from bridge_refiner import BoundaryRefiner
from bridge_vllm import BridgeVLLM, OptimizedPipelines
from bridge_bio import (
    GeneAnnotationManager, ExpressionPredictor, 
    CentralDogmaValidator, IntegratedBiologicalAnalysis
)
from bridge_api import BridgeApplication
from genome_handler import get_genome_handler

# Rich console for beautiful output
console = Console()

# ======================== Workflow Definitions ========================

class WorkflowOrchestrator:
    """
    Central orchestrator for all BRIDGE workflows
    """
    
    def __init__(self, config_path: str = "config/bridge_config.json"):
        self.config = self._load_config(config_path)
        self.console = console
        self.components = {}
        self.initialized = False
        
        # Available workflows
        self.workflows = {
            'single': self.single_sequence_analysis,
            'batch': self.batch_analysis,
            'region': self.genomic_region_analysis,
            'gene': self.gene_centric_analysis,
            'variant': self.variant_impact_analysis,
            'expression': self.expression_analysis,
            'multimodal': self.multimodal_analysis,
            'conservation': self.conservation_analysis,
            'design': self.regulatory_design,
            'network': self.regulatory_network_analysis,
            'benchmark': self.benchmark_analysis,
            'custom': self.custom_pipeline,
            'cross-species': self.cross_species_analysis,
            'evolution': self.evolutionary_analysis
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON or YAML"""
        path = Path(config_path)
        
        if not path.exists():
            self.console.print(f"[red]Config file not found: {config_path}[/red]")
            self.console.print("[yellow]Using default configuration[/yellow]")
            return self._get_default_config()
            
        with open(path, 'r') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                return yaml.safe_load(f)
            else:
                return json.load(f)
                
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            "models": {
                "use_hyenadna": True,
                "use_dnabert2": True,
                "use_lucaone": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "vllm": {
                "max_batch_tokens": 65536,
                "max_batch_size": 256,
                "batch_timeout": 0.1
            },
            "gene_annotation_file": "data/annotations/hg38.gtf",
            "genome": "hg38",
            "results_dir": "./results"
        }
        
    async def initialize(self):
        """Initialize all BRIDGE components"""
        if self.initialized:
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            # Model initialization
            task = progress.add_task("Initializing models...", total=3)
            
            self.components['model_manager'] = BridgeModelManager(self.config['models'])
            progress.update(task, advance=1)
            
            self.components['refiner'] = BoundaryRefiner(
                self.components['model_manager'].models
            )
            progress.update(task, advance=1)
            
            self.components['vllm'] = BridgeVLLM(
                self.components['model_manager'].models,
                self.config['vllm']
            )
            progress.update(task, advance=1)
            
            # Biological components
            bio_task = progress.add_task("Loading biological data...", total=3)
            
            self.components['gene_manager'] = GeneAnnotationManager(
                self.config['gene_annotation_file'],
                genome=self.config.get('genome', 'hg38')
            )
            progress.update(bio_task, advance=1)
            
            self.components['expression_predictor'] = ExpressionPredictor(
                self.components['model_manager']
            )
            progress.update(bio_task, advance=1)
            
            if 'lucaone' in self.components['model_manager'].models:
                self.components['dogma_validator'] = CentralDogmaValidator(
                    self.components['model_manager'].models['lucaone']
                )
            progress.update(bio_task, advance=1)
            
            self.components['bio_analyzer'] = IntegratedBiologicalAnalysis(
                self.components['gene_manager'],
                self.components['expression_predictor'],
                self.components.get('dogma_validator')
            )
            
            # Optimized pipelines
            self.components['pipelines'] = OptimizedPipelines(self.components['vllm'])
            
        self.initialized = True
        self.console.print("[green]✓ BRIDGE initialization complete[/green]")
        
    # ======================== Core Workflows ========================
    
    async def single_sequence_analysis(self, args: argparse.Namespace):
        """Analyze a single DNA sequence"""
        self.console.print(Panel.fit(
            f"[bold blue]Single Sequence Analysis[/bold blue]\n"
            f"Sequence length: {len(args.sequence)} bp"
        ))
        
        # Create sequence object
        seq = DNASequence(
            sequence=args.sequence,
            chromosome=args.chromosome,
            start=args.start,
            end=args.end
        )
        
        # Run analysis
        with self.console.status("Analyzing sequence..."):
            # Stage 1: Detect regulatory elements
            elements = await self.components['vllm'].analyze_sequences(
                [seq],
                task='regulatory_detection'
            )
            
            if elements and elements[0]:
                self.console.print(f"[green]Found {len(elements[0])} regulatory elements[/green]")
                
                # Stage 2: Refine boundaries
                refined = self.components['refiner'].batch_refine_elements(
                    elements[0],
                    [seq] * len(elements[0])
                )
                
                # Stage 3: Biological analysis
                if args.analyze_genes:
                    bio_results = self.components['bio_analyzer'].analyze_regulatory_landscape(
                        refined,
                        cell_type=args.cell_type or 'generic'
                    )
                    
                    self._display_biological_results(bio_results)
                else:
                    self._display_elements(refined)
            else:
                self.console.print("[yellow]No regulatory elements found[/yellow]")
                
    async def batch_analysis(self, args: argparse.Namespace):
        """Analyze multiple sequences from file"""
        self.console.print(Panel.fit(
            f"[bold blue]Batch Analysis[/bold blue]\n"
            f"Input file: {args.input_file}"
        ))
        
        # Load sequences
        sequences = self._load_sequences_from_file(args.input_file)
        self.console.print(f"Loaded {len(sequences)} sequences")
        
        # Process in batches
        results = []
        batch_size = args.batch_size or 100
        
        with Progress(console=self.console) as progress:
            task = progress.add_task(
                "Processing sequences...",
                total=len(sequences)
            )
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                # Analyze batch
                batch_results = await self.components['vllm'].analyze_sequences(
                    batch,
                    task='regulatory_detection',
                    priority=args.priority or 5
                )
                
                results.extend(batch_results)
                progress.update(task, advance=len(batch))
                
        # Save results
        self._save_batch_results(results, args.output or "batch_results.json")
        
        # Display summary
        self._display_batch_summary(results)
        
    async def genomic_region_analysis(self, args: argparse.Namespace):
        """Analyze a genomic region"""
        self.console.print(Panel.fit(
            f"[bold blue]Genomic Region Analysis[/bold blue]\n"
            f"Region: {args.chromosome}:{args.start:,}-{args.end:,}\n"
            f"Size: {args.end - args.start:,} bp"
        ))
        
        # Use optimized pipeline
        results = await self.components['pipelines'].full_locus_analysis(
            chromosome=args.chromosome,
            start=args.start,
            end=args.end,
            genome=args.genome or self.config['genome']
        )
        
        # Display results
        self._display_region_results(results)
        
        # Generate visualization if requested
        if args.visualize:
            self._visualize_region(results, args.output)
            
    async def gene_centric_analysis(self, args: argparse.Namespace):
        """Analyze regulatory landscape around a gene"""
        self.console.print(Panel.fit(
            f"[bold blue]Gene-Centric Analysis[/bold blue]\n"
            f"Gene: {args.gene_name}\n"
            f"Upstream: {args.upstream:,} bp\n"
            f"Downstream: {args.downstream:,} bp"
        ))
        
        # Find gene
        gene = self._find_gene_by_name(args.gene_name)
        if not gene:
            self.console.print(f"[red]Gene {args.gene_name} not found[/red]")
            return
            
        # Define analysis region
        if gene.strand == '+':
            region_start = gene.start - args.upstream
            region_end = gene.end + args.downstream
        else:
            region_start = gene.start - args.downstream
            region_end = gene.end + args.upstream
            
        # Analyze region
        results = await self.components['pipelines'].full_locus_analysis(
            chromosome=gene.chromosome,
            start=region_start,
            end=region_end
        )
        
        # Add gene context
        results['target_gene'] = {
            'gene_id': gene.gene_id,
            'gene_name': gene.gene_name,
            'type': gene.gene_type,
            'strand': gene.strand
        }
        
        # Predict expression
        if results['regulatory_elements']:
            expression = self.components['expression_predictor'].predict_expression_from_elements(
                gene,
                results['regulatory_elements'],
                cell_type=args.cell_type or 'generic'
            )
            results['predicted_expression'] = expression
            
        self._display_gene_analysis(results)
        
    async def variant_impact_analysis(self, args: argparse.Namespace):
        """Analyze regulatory impact of genetic variants"""
        self.console.print(Panel.fit(
            "[bold blue]Variant Impact Analysis[/bold blue]"
        ))
        
        # Load variants
        variants = self._load_variants(args.variants_file)
        self.console.print(f"Analyzing {len(variants)} variants")
        
        # Batch variant analysis
        results = await self.components['pipelines'].batch_variant_analysis(variants)
        
        # Display results
        affected = sum(1 for r in results if r['regulatory_impact']['affected'])
        self.console.print(f"\n[bold]Summary:[/bold]")
        self.console.print(f"Total variants: {len(variants)}")
        self.console.print(f"Affecting regulatory elements: {affected} ({affected/len(variants)*100:.1f}%)")
        
        # Detailed table
        if args.detailed:
            self._display_variant_table(results)
            
        # Save results
        if args.output:
            self._save_variant_results(results, args.output)
            
    async def expression_analysis(self, args: argparse.Namespace):
        """Predict gene expression from regulatory landscape"""
        self.console.print(Panel.fit(
            "[bold blue]Expression Analysis[/bold blue]"
        ))
        
        # Load gene list
        genes = self._load_gene_list(args.genes_file)
        
        results = {}
        with Progress(console=self.console) as progress:
            task = progress.add_task("Analyzing genes...", total=len(genes))
            
            for gene_name in genes:
                gene = self._find_gene_by_name(gene_name)
                if not gene:
                    continue
                    
                # Get regulatory elements around gene
                elements = await self._get_gene_regulatory_elements(gene)
                
                # Predict expression
                expression = self.components['expression_predictor'].predict_expression_from_elements(
                    gene,
                    elements,
                    cell_type=args.cell_type or 'generic'
                )
                
                results[gene_name] = {
                    'predicted_expression': expression,
                    'num_enhancers': sum(1 for e in elements if e.element_type == 'enhancer'),
                    'num_promoters': sum(1 for e in elements if e.element_type == 'promoter')
                }
                
                progress.update(task, advance=1)
                
        # Display results
        self._display_expression_results(results)
        
        # Generate heatmap if requested
        if args.heatmap:
            self._generate_expression_heatmap(results, args.output)
            
    async def multimodal_analysis(self, args: argparse.Namespace):
        """Multi-modal analysis using LucaOne"""
        self.console.print(Panel.fit(
            "[bold blue]Multi-Modal Analysis[/bold blue]\n"
            "Analyzing DNA-RNA-Protein relationships"
        ))
        
        if 'lucaone' not in self.components['model_manager'].models:
            self.console.print("[red]LucaOne model not available[/red]")
            return
            
        lucaone = self.components['model_manager'].models['lucaone']
        
        # Load sequences
        dna_seq = args.dna_sequence
        rna_seq = args.rna_sequence
        protein_seq = args.protein_sequence
        
        results = {}
        
        # DNA-Protein interaction
        if dna_seq and protein_seq:
            with self.console.status("Analyzing DNA-protein interaction..."):
                interaction_score = lucaone._verify_central_dogma(dna_seq, protein_seq)
                results['dna_protein_interaction'] = {
                    'score': interaction_score,
                    'valid': interaction_score > 0.8
                }
                
                # Find binding sites
                binding_sites = self._scan_binding_sites(dna_seq, protein_seq, lucaone)
                results['binding_sites'] = binding_sites
                
        # RNA-Protein interaction
        if rna_seq and protein_seq:
            with self.console.status("Analyzing RNA-protein interaction..."):
                rna_interaction = self._analyze_rna_protein(rna_seq, protein_seq, lucaone)
                results['rna_protein_interaction'] = rna_interaction
                
        # Central dogma validation
        if dna_seq and rna_seq and protein_seq:
            with self.console.status("Validating central dogma..."):
                cd_validation = self._validate_central_dogma_complete(
                    dna_seq, rna_seq, protein_seq, lucaone
                )
                results['central_dogma'] = cd_validation
                
        self._display_multimodal_results(results)
        
    async def conservation_analysis(self, args: argparse.Namespace):
        """Analyze regulatory conservation across species"""
        self.console.print(Panel.fit(
            "[bold blue]Conservation Analysis[/bold blue]\n"
            f"Reference species: {args.reference_species}\n"
            f"Target species: {', '.join(args.target_species)}"
        ))
        
        # This would integrate with conservation databases
        self.console.print("[yellow]Conservation analysis requires external databases[/yellow]")
        self.console.print("This is a placeholder for the full implementation")
        
    async def regulatory_design(self, args: argparse.Namespace):
        """Design synthetic regulatory elements"""
        self.console.print(Panel.fit(
            "[bold blue]Regulatory Element Design[/bold blue]\n"
            f"Target genes: {', '.join(args.target_genes)}\n"
            f"Desired expression: {args.expression_level}x"
        ))
        
        # This would use generative models
        self.console.print("[yellow]Regulatory design is in development[/yellow]")
        
        # Placeholder design
        designed_element = {
            'sequence': 'ATCGATCG' * 50,  # Placeholder
            'predicted_activity': args.expression_level,
            'target_genes': args.target_genes,
            'confidence': 0.75
        }
        
        self._display_designed_element(designed_element)
        
    async def regulatory_network_analysis(self, args: argparse.Namespace):
        """Analyze regulatory networks"""
        self.console.print(Panel.fit(
            "[bold blue]Regulatory Network Analysis[/bold blue]"
        ))
        
        # Load gene list
        genes = self._load_gene_list(args.genes_file)
        
        # Build network
        network = await self._build_regulatory_network(genes)
        
        # Display network statistics
        self._display_network_stats(network)
        
        # Export if requested
        if args.export_format:
            self._export_network(network, args.output, args.export_format)
            
    async def benchmark_analysis(self, args: argparse.Namespace):
        """Benchmark BRIDGE performance"""
        self.console.print(Panel.fit(
            "[bold blue]Performance Benchmark[/bold blue]"
        ))
        
        # Generate test sequences
        test_sequences = self._generate_benchmark_sequences(args.num_sequences)
        
        # Benchmark different components
        results = {}
        
        # Test regulatory detection
        start_time = time.time()
        await self.components['vllm'].analyze_sequences(
            test_sequences[:10],
            task='regulatory_detection'
        )
        results['regulatory_detection_time'] = time.time() - start_time
        
        # Test throughput
        throughput_results = await self._benchmark_throughput(test_sequences)
        results['throughput'] = throughput_results
        
        # Display results
        self._display_benchmark_results(results)
        
    async def custom_pipeline(self, args: argparse.Namespace):
        """Run custom analysis pipeline from config file"""
        self.console.print(Panel.fit(
            f"[bold blue]Custom Pipeline[/bold blue]\n"
            f"Config: {args.pipeline_config}"
        ))
        
        # Load pipeline configuration
        with open(args.pipeline_config, 'r') as f:
            pipeline_config = yaml.safe_load(f)
            
        # Execute pipeline steps
        results = {}
        for step in pipeline_config['steps']:
            self.console.print(f"\n[bold]Step: {step['name']}[/bold]")
            
            if step['type'] == 'regulatory_detection':
                step_results = await self._run_regulatory_step(step)
            elif step['type'] == 'expression_prediction':
                step_results = await self._run_expression_step(step)
            elif step['type'] == 'multimodal':
                step_results = await self._run_multimodal_step(step)
            else:
                self.console.print(f"[red]Unknown step type: {step['type']}[/red]")
                continue
                
            results[step['name']] = step_results
            
        # Save final results
        self._save_pipeline_results(results, args.output or "pipeline_results.json")
        
    async def cross_species_analysis(self, args: argparse.Namespace):
        """
        Analyze regulatory conservation across species
        Leverages HyenaDNA's mean pooling for cross-species generalization
        """
        self.console.print(Panel.fit(
            f"[bold blue]Cross-Species Analysis[/bold blue]\n"
            f"Reference: {args.reference_species}\n"
            f"Target species: {', '.join(args.target_species)}"
        ))
        
        # Load sequences for each species
        all_sequences = {}
        for species in [args.reference_species] + args.target_species:
            if args.input_format == 'coordinates':
                # Get orthologous regions
                seq = await self._get_orthologous_sequence(
                    args.chromosome, args.start, args.end, 
                    args.reference_species, species
                )
            else:
                # Load from file
                seq = self._load_species_sequence(args.input_file, species)
                
            if seq:
                all_sequences[species] = seq
            else:
                self.console.print(f"[yellow]Warning: Could not load sequence for {species}[/yellow]")
                
        # Analyze each species
        all_elements = {}
        for species, sequence in all_sequences.items():
            self.console.print(f"\nAnalyzing {species}...")
            
            # Create DNASequence object with species info
            dna_seq = DNASequence(
                sequence=sequence,
                chromosome=args.chromosome,
                start=args.start,
                end=args.end
            )
            
            # Run standard BRIDGE analysis
            # Works across species due to mean pooling!
            elements = await self.components['vllm'].analyze_sequences(
                [dna_seq],
                task='regulatory_detection'
            )
            
            if elements and elements[0]:
                all_elements[species] = elements[0]
                self.console.print(f"Found {len(elements[0])} elements in {species}")
                
        # Compare across species
        if len(all_elements) > 1:
            self.console.print("\n[bold]Cross-species comparison:[/bold]")
            conservation_results = self._compare_cross_species_elements(
                all_elements
            )
            
            # Display conserved elements
            self._display_conservation_results(conservation_results)
            
            # Save results
            if args.output:
                self._save_cross_species_results(
                    all_elements, conservation_results, args.output
                )
                
    async def evolutionary_analysis(self, args: argparse.Namespace):
        """
        Analyze regulatory evolution of genes across species
        Uses mean-pooled features to track conservation
        """
        self.console.print(Panel.fit(
            f"[bold blue]Evolutionary Regulatory Analysis[/bold blue]\n"
            f"Genes: {', '.join(args.gene_list)}\n"
            f"Species: {', '.join(args.species_list)}"
        ))
        
        results = {}
        
        for gene_name in args.gene_list:
            self.console.print(f"\n[bold]Analyzing {gene_name}[/bold]")
            gene_results = {'orthologs': {}, 'regulatory_evolution': {}}
            
            for species in args.species_list:
                # Find ortholog
                ortholog = await self._find_gene_ortholog(gene_name, 'human', species)
                
                if ortholog:
                    # Analyze regulatory landscape
                    region_start = ortholog['start'] - args.upstream
                    region_end = ortholog['end'] + args.downstream
                    
                    seq = await self._fetch_sequence(
                        ortholog['chromosome'], region_start, region_end, species
                    )
                    
                    # Analyze with BRIDGE
                    dna_seq = DNASequence(
                        sequence=seq,
                        chromosome=ortholog['chromosome'],
                        start=region_start,
                        end=region_end
                    )
                    
                    elements = await self.components['vllm'].analyze_sequences(
                        [dna_seq],
                        task='regulatory_detection'
                    )
                    
                    gene_results['orthologs'][species] = {
                        'gene_info': ortholog,
                        'regulatory_elements': elements[0] if elements else [],
                        'enhancer_count': sum(1 for e in elements[0] if e.element_type == 'enhancer') if elements else 0
                    }
                    
            # Analyze evolutionary patterns
            if len(gene_results['orthologs']) > 1:
                gene_results['regulatory_evolution'] = self._analyze_regulatory_evolution(
                    gene_results['orthologs']
                )
                
            results[gene_name] = gene_results
            
        # Display summary
        self._display_evolutionary_summary(results)
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
    # ======================== Helper Methods ========================
    
    async def _get_orthologous_sequence(self, chromosome: str, start: int, end: int,
                                      source_species: str, target_species: str) -> Optional[str]:
        """Get orthologous sequence from another species"""
        genome_handler = get_genome_handler()
        
        # Get orthologous sequence
        ortho_seq = genome_handler.get_orthologous_sequence(
            chromosome, start, end,
            source_species, target_species,
            extend_flanks=True
        )
        
        if ortho_seq:
            return ortho_seq
        else:
            self.console.print(f"[yellow]Could not find orthologous sequence in {target_species}[/yellow]")
            return None

    async def _find_gene_ortholog(self, gene_name: str, source_species: str, 
                                target_species: str) -> Optional[Dict]:
        """Find orthologous gene in target species"""
        genome_handler = get_genome_handler()
        
        # First find the gene in source species
        source_gene = None
        for gene in self.components['gene_manager'].genes.values():
            if gene.gene_name == gene_name:
                source_gene = gene
                break
                
        if not source_gene:
            return None
            
        # Find orthologs
        orthologs = genome_handler.find_orthologs(
            source_gene.gene_id,
            source_species,
            target_species
        )
        
        if orthologs:
            # Get best ortholog
            best_ortholog = max(orthologs, key=lambda x: x.confidence)
            
            # Get target gene info
            target_gene_info = {
                'gene_id': best_ortholog.target_gene_id,
                'gene_name': gene_name + f"_{target_species}",  # Placeholder
                'chromosome': None,
                'start': None,
                'end': None,
                'confidence': best_ortholog.confidence
            }
            
            # Would need to query target species gene database for full info
            return target_gene_info
        
        return None

    def _analyze_regulatory_evolution(self, orthologs: Dict[str, Dict]) -> Dict:
        """Analyze regulatory evolution across species"""
        evolution_results = {
            'ultra_conserved': [],
            'lineage_specific': defaultdict(list),
            'conservation_scores': {},
            'regulatory_turnover': {}
        }
        
        # Get all regulatory elements across species
        all_elements_by_species = {}
        for species, data in orthologs.items():
            if 'regulatory_elements' in data:
                all_elements_by_species[species] = data['regulatory_elements']
                
        # Find ultra-conserved elements (present in all species)
        if len(all_elements_by_species) > 1:
            # Simple approach: elements at similar relative positions
            reference_species = list(all_elements_by_species.keys())[0]
            reference_elements = all_elements_by_species[reference_species]
            
            for ref_elem in reference_elements:
                conserved_count = 1
                conservation_data = {reference_species: ref_elem}
                
                # Check other species
                for species, elements in all_elements_by_species.items():
                    if species == reference_species:
                        continue
                        
                    # Find matching element
                    for elem in elements:
                        # Check if elements overlap in relative position
                        if (elem.element_type == ref_elem.element_type and
                            abs(elem.start - ref_elem.start) < 100):  # Within 100bp
                            conserved_count += 1
                            conservation_data[species] = elem
                            break
                            
                # If conserved in >80% of species, consider ultra-conserved
                if conserved_count / len(all_elements_by_species) > 0.8:
                    evolution_results['ultra_conserved'].append({
                        'type': ref_elem.element_type,
                        'conservation': conserved_count / len(all_elements_by_species),
                        'elements': conservation_data
                    })
                    
        # Find lineage-specific elements
        for species, elements in all_elements_by_species.items():
            for elem in elements:
                # Check if element is unique to this species
                is_unique = True
                
                for other_species, other_elements in all_elements_by_species.items():
                    if other_species == species:
                        continue
                        
                    for other_elem in other_elements:
                        if (elem.element_type == other_elem.element_type and
                            abs(elem.start - other_elem.start) < 100):
                            is_unique = False
                            break
                            
                    if not is_unique:
                        break
                        
                if is_unique:
                    evolution_results['lineage_specific'][species].append(elem)
                    
        # Calculate conservation scores
        for species in all_elements_by_species:
            if species in orthologs and 'enhancer_count' in orthologs[species]:
                evolution_results['conservation_scores'][species] = {
                    'enhancer_count': orthologs[species]['enhancer_count'],
                    'total_elements': len(all_elements_by_species.get(species, []))
                }
                
        # Calculate regulatory turnover rate
        if len(all_elements_by_species) > 1:
            species_list = list(all_elements_by_species.keys())
            for i in range(len(species_list) - 1):
                sp1, sp2 = species_list[i], species_list[i + 1]
                
                elems1 = set((e.element_type, e.start // 100) for e in all_elements_by_species[sp1])
                elems2 = set((e.element_type, e.start // 100) for e in all_elements_by_species[sp2])
                
                shared = len(elems1 & elems2)
                total = len(elems1 | elems2)
                
                if total > 0:
                    evolution_results['regulatory_turnover'][f"{sp1}_to_{sp2}"] = {
                        'shared': shared,
                        'total': total,
                        'turnover_rate': 1 - (shared / total)
                    }
                    
        return evolution_results

    def _compare_cross_species_elements(self, all_elements: Dict[str, List[RegulatoryElement]]) -> Dict:
        """Compare regulatory elements across species"""
        # Simplified version of _analyze_regulatory_evolution
        return self._analyze_regulatory_evolution({'species': {'regulatory_elements': elems} 
                                                  for species, elems in all_elements.items()})
    
    def _load_sequences_from_file(self, filepath: str) -> List[DNASequence]:
        """Load sequences from FASTA/text file"""
        sequences = []
        
        if filepath.endswith('.fasta') or filepath.endswith('.fa'):
            from Bio import SeqIO
            for record in SeqIO.parse(filepath, "fasta"):
                sequences.append(DNASequence(
                    sequence=str(record.seq),
                    chromosome=record.id
                ))
        else:
            # Plain text, one sequence per line
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        sequences.append(DNASequence(
                            sequence=line.strip(),
                            chromosome=f"seq_{i}"
                        ))
                        
        return sequences
        
    def _find_gene_by_name(self, gene_name: str):
        """Find gene by name"""
        for gene in self.components['gene_manager'].genes.values():
            if gene.gene_name == gene_name:
                return gene
        return None
        
    def _load_gene_list(self, filepath: str) -> List[str]:
        """Load gene list from file"""
        genes = []
        
        with open(filepath, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('#'):
                    genes.append(gene)
                    
        return genes

    async def _get_gene_regulatory_elements(self, gene) -> List[RegulatoryElement]:
        """Get regulatory elements around a gene"""
        # Define search region
        search_start = gene.tss - 100000  # 100kb upstream
        search_end = gene.tss + 10000     # 10kb downstream
        
        # Create sequence object
        seq = DNASequence(
            sequence=await self._fetch_sequence(
                gene.chromosome, search_start, search_end
            ),
            chromosome=gene.chromosome,
            start=search_start,
            end=search_end
        )
        
        # Analyze
        elements = await self.components['vllm'].analyze_sequences(
            [seq],
            task='regulatory_detection'
        )
        
        return elements[0] if elements else []

    async def _fetch_sequence(self, chromosome: str, start: int, end: int,
                            genome: str = 'hg38') -> str:
        """Fetch genomic sequence"""
        genome_handler = get_genome_handler()
        return genome_handler.fetch_sequence(chromosome, start, end, genome)

    def _load_variants(self, variants_file: str) -> List[Dict]:
        """Load variants from VCF or TSV file"""
        variants = []
        
        if variants_file.endswith('.vcf'):
            # Parse VCF format
            with open(variants_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                        
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        variants.append({
                            'chrom': parts[0],
                            'pos': int(parts[1]),
                            'ref': parts[3],
                            'alt': parts[4]
                        })
        else:
            # Assume TSV format
            df = pd.read_csv(variants_file, sep='\t')
            for _, row in df.iterrows():
                variants.append({
                    'chrom': row['chrom'],
                    'pos': row['pos'],
                    'ref': row['ref'],
                    'alt': row['alt']
                })
                
        return variants

    def _load_species_sequence(self, input_file: str, species: str) -> Optional[str]:
        """Load species-specific sequence from file"""
        try:
            # Assuming file format: species_sequences.fasta
            from Bio import SeqIO
            
            for record in SeqIO.parse(input_file, "fasta"):
                if species in record.id or species in record.description:
                    return str(record.seq).upper()
                    
            return None
        except Exception as e:
            self.console.print(f"[red]Error loading sequence for {species}: {e}[/red]")
            return None

    def _scan_binding_sites(self, dna_seq: str, protein_seq: str, lucaone) -> List[Dict]:
        """Scan for potential DNA-protein binding sites"""
        binding_sites = []
        window_size = 20
        stride = 5
        
        for i in range(0, len(dna_seq) - window_size, stride):
            window = dna_seq[i:i + window_size]
            
            score = lucaone._verify_central_dogma(window, protein_seq)
            
            if score > 0.7:
                binding_sites.append({
                    'position': i,
                    'score': score,
                    'sequence': window
                })
                
        return binding_sites

    def _analyze_rna_protein(self, rna_seq: str, protein_seq: str, lucaone) -> Dict:
        """Analyze RNA-protein interaction"""
        # Simple analysis using LucaOne
        rna_emb = lucaone._encode_sequence(rna_seq, 'RNA')
        prot_emb = lucaone._encode_sequence(protein_seq, 'PROTEIN')
        
        similarity = torch.cosine_similarity(rna_emb, prot_emb, dim=-1).item()
        
        return {
            'interaction_score': similarity,
            'likely_interaction': similarity > 0.7
        }

    def _validate_central_dogma_complete(self, dna_seq: str, rna_seq: str, 
                                       protein_seq: str, lucaone) -> Dict:
        """Complete central dogma validation"""
        return {
            'dna_rna_valid': lucaone._verify_central_dogma(dna_seq, rna_seq) > 0.8,
            'rna_protein_valid': lucaone._verify_central_dogma(rna_seq, protein_seq) > 0.8,
            'dna_protein_valid': lucaone._verify_central_dogma(dna_seq, protein_seq) > 0.8
        }

    def _display_elements(self, elements: List[RegulatoryElement]):
        """Display regulatory elements in a table"""
        table = Table(title="Regulatory Elements Found")
        table.add_column("Type", style="cyan")
        table.add_column("Location", style="magenta")
        table.add_column("Length", justify="right", style="green")
        table.add_column("Score", justify="right", style="yellow")
        table.add_column("TF Motifs", style="blue")
        
        for elem in elements:
            location = f"{elem.chr}:{elem.start:,}-{elem.end:,}"
            tf_motifs = ", ".join([m['tf'] for m in elem.tf_motifs[:3]])
            if len(elem.tf_motifs) > 3:
                tf_motifs += f" (+{len(elem.tf_motifs)-3} more)"
                
            table.add_row(
                elem.element_type.capitalize(),
                location,
                f"{elem.length} bp",
                f"{elem.score:.3f}",
                tf_motifs or "None"
            )
            
        self.console.print(table)
        
    def _display_biological_results(self, results: Dict):
        """Display biological analysis results"""
        # Gene assignments
        self.console.print("\n[bold]Gene Assignments:[/bold]")
        gene_table = Table()
        gene_table.add_column("Gene", style="cyan")
        gene_table.add_column("Distance to TSS", justify="right")
        gene_table.add_column("Interaction Type")
        gene_table.add_column("Confidence", justify="right")
        
        for interaction in results['gene_assignments'][:10]:  # Top 10
            gene_table.add_row(
                interaction.gene.gene_name,
                f"{interaction.distance:,} bp",
                interaction.interaction_type,
                f"{interaction.confidence:.2f}"
            )
            
        self.console.print(gene_table)
        
        # Expression predictions
        if results['expression_predictions']:
            self.console.print("\n[bold]Expression Predictions:[/bold]")
            expr_table = Table()
            expr_table.add_column("Gene", style="cyan")
            expr_table.add_column("Predicted Expression", justify="right")
            expr_table.add_column("Regulatory Elements", justify="right")
            
            for gene_id, expr_data in list(results['expression_predictions'].items())[:10]:
                expr_table.add_row(
                    expr_data['gene_name'],
                    f"{expr_data['predicted_expression']:.2f}",
                    str(expr_data['num_regulatory_elements'])
                )
                
            self.console.print(expr_table)
            
        # Summary statistics
        if results['summary_statistics']:
            self.console.print("\n[bold]Summary Statistics:[/bold]")
            stats = results['summary_statistics']
            tree = Tree("Analysis Summary")
            tree.add(f"Total genes analyzed: {stats.get('total_genes', 0)}")
            tree.add(f"Total regulatory elements: {stats.get('total_elements', 0)}")
            tree.add(f"Elements per gene: {stats.get('elements_per_gene', 0):.2f}")
            tree.add(f"Mean validation score: {stats.get('mean_validation_score', 0):.3f}")
            self.console.print(tree)

    def _save_batch_results(self, results: List, output_path: str):
        """Save batch analysis results"""
        # Convert to serializable format
        output_data = []
        for i, elem_list in enumerate(results):
            seq_data = {
                'sequence_index': i,
                'elements': []
            }
            
            if elem_list:  # Check if not empty
                for elem in elem_list:
                    seq_data['elements'].append({
                        'type': elem.element_type,
                        'start': elem.start,
                        'end': elem.end,
                        'score': elem.score,
                        'length': elem.length
                    })
                
            output_data.append(seq_data)
            
        # Save based on format
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        elif output_path.endswith('.tsv'):
            # Flatten for TSV
            rows = []
            for seq_data in output_data:
                for elem in seq_data['elements']:
                    row = {
                        'sequence_index': seq_data['sequence_index'],
                        **elem
                    }
                    rows.append(row)
                    
            df = pd.DataFrame(rows)
            df.to_csv(output_path, sep='\t', index=False)
            
        self.console.print(f"[green]Results saved to {output_path}[/green]")

    def _display_batch_summary(self, results: List):
        """Display batch analysis summary"""
        total_elements = sum(len(r) if r else 0 for r in results)
        sequences_with_elements = sum(1 for r in results if r)
        
        self.console.print("\n[bold]Batch Analysis Summary[/bold]")
        self.console.print(f"Total sequences analyzed: {len(results)}")
        self.console.print(f"Sequences with regulatory elements: {sequences_with_elements}")
        self.console.print(f"Total regulatory elements found: {total_elements}")
        
        if total_elements > 0:
            # Element type breakdown
            element_types = defaultdict(int)
            for result in results:
                if result:
                    for elem in result:
                        element_types[elem.element_type] += 1
                        
            self.console.print("\n[bold]Element Type Distribution:[/bold]")
            for elem_type, count in element_types.items():
                self.console.print(f"  {elem_type.capitalize()}: {count}")

    def _display_region_results(self, results: Dict):
        """Display genomic region analysis results"""
        self.console.print(f"\n[bold]Region Analysis Results[/bold]")
        self.console.print(f"Locus: {results['locus']}")
        
        if 'regulatory_elements' in results:
            self._display_elements(results['regulatory_elements'])
            
        if 'summary' in results:
            summary = results['summary']
            self.console.print(f"\n[bold]Summary:[/bold]")
            self.console.print(f"Total elements: {summary.get('total_elements', 0)}")
            self.console.print(f"Enhancers: {summary.get('enhancers', 0)}")
            self.console.print(f"Promoters: {summary.get('promoters', 0)}")

    def _visualize_region(self, results: Dict, output_path: Optional[str]):
        """Create visualization of genomic region with regulatory elements"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Extract data
        elements = results.get('regulatory_elements', [])
        if not elements:
            self.console.print("[yellow]No elements to visualize[/yellow]")
            return
            
        # Determine region bounds
        chr_name = elements[0].chr
        start = min(e.start for e in elements) - 1000
        end = max(e.end for e in elements) + 1000
        
        # Axis 1: Regulatory elements
        ax1 = axes[0]
        ax1.set_title(f'Regulatory Elements - {chr_name}:{start:,}-{end:,}', fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_ylabel('Elements')
        
        # Color map for element types
        colors = {'enhancer': '#e74c3c', 'promoter': '#3498db', 'silencer': '#95a5a6'}
        
        for elem in elements:
            color = colors.get(elem.element_type, '#34495e')
            # Draw element
            ax1.barh(0, elem.end - elem.start, left=elem.start, height=0.6,
                    color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add label if space permits
            if elem.end - elem.start > (end - start) * 0.05:
                ax1.text((elem.start + elem.end) / 2, 0, elem.element_type,
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
                        
        ax1.set_yticks([0])
        ax1.set_yticklabels([''])
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=etype.capitalize())
                          for etype, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Axis 2: Element scores
        ax2 = axes[1]
        ax2.set_title('Regulatory Scores', fontsize=12)
        ax2.set_ylabel('Score')
        
        # Create score profile
        x_positions = []
        y_scores = []
        
        for elem in elements:
            # Add points for element boundaries and center
            x_positions.extend([elem.start, (elem.start + elem.end) / 2, elem.end])
            y_scores.extend([0, elem.score, 0])
            
        if x_positions:
            ax2.fill_between(x_positions, y_scores, alpha=0.4, color='#3498db')
            ax2.plot(x_positions, y_scores, color='#2c3e50', linewidth=2)
            
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Axis 3: Conservation (if available)
        ax3 = axes[2]
        ax3.set_title('Conservation Score', fontsize=12)
        ax3.set_xlabel('Genomic Position')
        ax3.set_ylabel('PhyloP')
        
        # Get conservation scores
        genome_handler = get_genome_handler()
        try:
            conservation = genome_handler.get_conservation_score(
                chr_name, start, end, genome='hg38'
            )
            
            positions = np.arange(start, end)
            ax3.plot(positions, conservation, color='#27ae60', linewidth=1, alpha=0.8)
            ax3.fill_between(positions, conservation, alpha=0.3, color='#27ae60')
            
            # Highlight conserved regions
            threshold = np.percentile(conservation[conservation > 0], 75)
            ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'75th percentile')
            ax3.legend()
            
        except Exception as e:
            ax3.text(0.5, 0.5, 'Conservation data not available', 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(start, end)
        
        # Format x-axis
        ax3.ticklabel_format(style='plain', axis='x')
        ax3.set_xticks(np.linspace(start, end, 5))
        ax3.set_xticklabels([f'{int(x):,}' for x in ax3.get_xticks()], rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.console.print(f"[green]✓ Visualization saved to {output_path}[/green]")
        else:
            plt.show()

    def _display_gene_analysis(self, results: Dict):
        """Display gene-centric analysis results"""
        # Create summary panel
        gene_info = results.get('target_gene', {})
        
        summary_text = f"""
[bold]Gene:[/bold] {gene_info.get('gene_name', 'Unknown')}
[bold]Type:[/bold] {gene_info.get('type', 'Unknown')}
[bold]Location:[/bold] {results.get('locus', 'Unknown')}
[bold]Strand:[/bold] {gene_info.get('strand', '?')}

[bold]Regulatory Elements Found:[/bold]
  Total: {len(results.get('regulatory_elements', []))}
  Enhancers: {sum(1 for e in results.get('regulatory_elements', []) if e.element_type == 'enhancer')}
  Promoters: {sum(1 for e in results.get('regulatory_elements', []) if e.element_type == 'promoter')}

[bold]Predicted Expression:[/bold] {results.get('predicted_expression', 'N/A'):.2f}
        """
        
        self.console.print(Panel(summary_text.strip(), title="Gene Analysis Summary"))
        
        # Show top elements
        if results.get('regulatory_elements'):
            self._display_elements(results['regulatory_elements'][:10])

    def _display_variant_table(self, results: List[Dict]):
        """Display variant analysis results in a table"""
        table = Table(title="Variant Impact Analysis")
        table.add_column("Variant", style="cyan")
        table.add_column("Affects Regulatory", style="magenta")
        table.add_column("Element Type", style="green")
        table.add_column("Disrupted Motifs", style="yellow")
        
        for result in results[:20]:  # Show first 20
            variant = result['variant']
            impact = result['regulatory_impact']
            
            variant_str = f"{variant['chrom']}:{variant['pos']} {variant['ref']}>{variant['alt']}"
            
            if impact['affected']:
                affects = "Yes"
                elem_type = impact['element'].element_type
                motifs = len(impact.get('disrupted_motifs', []))
            else:
                affects = "No"
                elem_type = "-"
                motifs = 0
                
            table.add_row(variant_str, affects, elem_type, str(motifs))
            
        self.console.print(table)

    def _save_variant_results(self, results: List[Dict], output_path: str):
        """Save variant impact analysis results"""
        output_path = Path(output_path)
        
        # Convert to DataFrame for easy manipulation
        data = []
        for result in results:
            variant = result['variant']
            impact = result['regulatory_impact']
            
            row = {
                'chrom': variant['chrom'],
                'pos': variant['pos'],
                'ref': variant['ref'],
                'alt': variant['alt'],
                'affects_regulatory': impact['affected']
            }
            
            if impact['affected']:
                elem = impact['element']
                row.update({
                    'element_type': elem.element_type,
                    'element_start': elem.start,
                    'element_end': elem.end,
                    'position_in_element': impact['position_in_element'],
                    'disrupted_motifs': len(impact.get('disrupted_motifs', []))
                })
                
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Save based on file extension
        if output_path.suffix == '.tsv':
            df.to_csv(output_path, sep='\t', index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_path, index=False)
            
        self.console.print(f"[green]✓ Variant results saved to {output_path}[/green]")

    def _display_expression_results(self, results: Dict):
        """Display expression analysis results"""
        self.console.print("\n[bold]Expression Prediction Results[/bold]\n")
        
        # Sort by expression level
        sorted_genes = sorted(results.items(), 
                             key=lambda x: x[1]['predicted_expression'], 
                             reverse=True)
        
        table = Table()
        table.add_column("Gene", style="cyan")
        table.add_column("Predicted Expression", justify="right", style="green")
        table.add_column("Enhancers", justify="right", style="yellow")
        table.add_column("Promoters", justify="right", style="magenta")
        
        for gene_name, data in sorted_genes[:20]:  # Top 20
            table.add_row(
                gene_name,
                f"{data['predicted_expression']:.2f}",
                str(data['num_enhancers']),
                str(data['num_promoters'])
            )
            
        self.console.print(table)

    def _generate_expression_heatmap(self, results: Dict, output_path: Optional[str]):
        """Generate heatmap of predicted expression across genes"""
        # Convert results to DataFrame
        data = []
        for gene_name, expr_data in results.items():
            data.append({
                'Gene': gene_name,
                'Expression': expr_data['predicted_expression'],
                'Enhancers': expr_data['num_enhancers'],
                'Promoters': expr_data['num_promoters']
            })
            
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            self.console.print("[yellow]No expression data to visualize[/yellow]")
            return
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.3)))
        
        # Expression heatmap
        expr_matrix = df[['Expression']].values.T
        
        sns.heatmap(expr_matrix, 
                    xticklabels=df['Gene'],
                    yticklabels=['Expression'],
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Predicted Expression (log2)'},
                    ax=ax1)
        
        ax1.set_title('Predicted Gene Expression', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Regulatory element counts
        element_data = df[['Enhancers', 'Promoters']].set_index(df['Gene'])
        
        element_data.plot(kind='barh', ax=ax2, color=['#e74c3c', '#3498db'])
        ax2.set_xlabel('Number of Elements')
        ax2.set_title('Regulatory Elements per Gene', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.console.print(f"[green]✓ Heatmap saved to {output_path}[/green]")
        else:
            plt.show()

    def _display_multimodal_results(self, results: Dict):
        """Display multi-modal analysis results"""
        if 'dna_protein_interaction' in results:
            interaction = results['dna_protein_interaction']
            self.console.print("\n[bold]DNA-Protein Interaction:[/bold]")
            self.console.print(f"Interaction score: {interaction['score']:.3f}")
            self.console.print(f"Valid interaction: {'✓' if interaction['valid'] else '✗'}")
            
        if 'binding_sites' in results:
            self.console.print("\n[bold]Predicted Binding Sites:[/bold]")
            table = Table()
            table.add_column("Position", justify="right")
            table.add_column("Score", justify="right")
            table.add_column("Sequence")
            
            for site in results['binding_sites'][:5]:  # Top 5
                table.add_row(
                    str(site['position']),
                    f"{site['score']:.3f}",
                    site['sequence']
                )
                
            self.console.print(table)
            
        if 'central_dogma' in results:
            cd = results['central_dogma']
            self.console.print("\n[bold]Central Dogma Validation:[/bold]")
            tree = Tree("Central Dogma")
            tree.add(f"DNA → RNA: {'✓' if cd.get('dna_rna_valid', False) else '✗'}")
            tree.add(f"RNA → Protein: {'✓' if cd.get('rna_protein_valid', False) else '✗'}")
            tree.add(f"DNA → Protein: {'✓' if cd.get('dna_protein_valid', False) else '✗'}")
            self.console.print(tree)

    def _display_designed_element(self, element: Dict):
        """Display designed regulatory element"""
        design_text = f"""
[bold]Designed Regulatory Element[/bold]

[bold]Sequence:[/bold]
{element['sequence'][:50]}...{element['sequence'][-50:]}
Length: {len(element['sequence'])} bp

[bold]Predicted Activity:[/bold] {element['predicted_activity']}x
[bold]Target Genes:[/bold] {', '.join(element['target_genes'])}
[bold]Confidence:[/bold] {element['confidence']:.2f}
        """
        
        self.console.print(Panel(design_text.strip(), title="Regulatory Design Result"))

    async def _build_regulatory_network(self, genes: List[str]) -> List[Dict]:
        """Build regulatory network for gene list"""
        network = []
        
        # Find TF genes
        tf_genes = [g for g in genes if g in self.components['dogma_validator']._get_tf_list()]
        
        for tf in tf_genes:
            tf_targets = []
            
            # Get binding sites for this TF
            for target_gene in genes:
                if target_gene == tf:
                    continue
                    
                # Get regulatory elements for target gene
                gene_obj = self._find_gene_by_name(target_gene)
                if not gene_obj:
                    continue
                    
                elements = await self._get_gene_regulatory_elements(gene_obj)
                
                # Check for TF binding
                for elem in elements:
                    for motif in elem.tf_motifs:
                        if motif['tf'] == tf:
                            tf_targets.append({
                                'target_gene': target_gene,
                                'element': f"{elem.chr}:{elem.start}-{elem.end}",
                                'binding_score': motif['score']
                            })
                            break
                            
            if tf_targets:
                network.append({
                    'tf': tf,
                    'targets': tf_targets
                })
                
        return network

    def _display_network_stats(self, network: List[Dict]):
        """Display network statistics"""
        self.console.print("\n[bold]Regulatory Network Statistics[/bold]\n")
        
        # Calculate stats
        total_tfs = len(network)
        total_edges = sum(len(n['targets']) for n in network)
        
        # Find hub TFs
        hub_tfs = sorted(network, key=lambda x: len(x['targets']), reverse=True)[:5]
        
        tree = Tree("Network Summary")
        tree.add(f"Transcription factors: {total_tfs}")
        tree.add(f"Regulatory interactions: {total_edges}")
        
        if hub_tfs:
            hub_branch = tree.add("Hub TFs (most targets)")
            for tf_data in hub_tfs:
                hub_branch.add(f"{tf_data['tf']}: {len(tf_data['targets'])} targets")
                
        self.console.print(tree)

    def _export_network(self, network: List[Dict], output_path: str, format: str):
        """Export regulatory network in various formats"""
        output_path = Path(output_path)
        
        if format == 'json':
            # Simple JSON export
            with open(output_path, 'w') as f:
                json.dump(network, f, indent=2)
                
        elif format == 'graphml':
            # GraphML format for Cytoscape
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes and edges
            for tf_data in network:
                tf = tf_data['tf']
                G.add_node(tf, node_type='TF')
                
                for target in tf_data['targets']:
                    target_gene = target['target_gene']
                    G.add_node(target_gene, node_type='gene')
                    G.add_edge(tf, target_gene, 
                              element=target['element'],
                              score=target['binding_score'])
                              
            nx.write_graphml(G, output_path)
            
        elif format == 'cytoscape':
            # Cytoscape JSON format
            cytoscape_data = {
                'elements': {
                    'nodes': [],
                    'edges': []
                }
            }
            
            node_ids = set()
            
            for tf_data in network:
                tf = tf_data['tf']
                
                # Add TF node
                if tf not in node_ids:
                    cytoscape_data['elements']['nodes'].append({
                        'data': {'id': tf, 'label': tf, 'type': 'TF'}
                    })
                    node_ids.add(tf)
                    
                # Add targets and edges
                for target in tf_data['targets']:
                    target_gene = target['target_gene']
                    
                    if target_gene not in node_ids:
                        cytoscape_data['elements']['nodes'].append({
                            'data': {'id': target_gene, 'label': target_gene, 'type': 'gene'}
                        })
                        node_ids.add(target_gene)
                        
                    # Add edge
                    cytoscape_data['elements']['edges'].append({
                        'data': {
                            'source': tf,
                            'target': target_gene,
                            'score': target['binding_score']
                        }
                    })
                    
            with open(output_path, 'w') as f:
                json.dump(cytoscape_data, f, indent=2)
                
        self.console.print(f"[green]✓ Network exported to {output_path} ({format} format)[/green]")

    def _generate_benchmark_sequences(self, num_sequences: int) -> List[DNASequence]:
        """Generate synthetic sequences for benchmarking"""
        sequences = []
        
        # Different sequence lengths for testing
        length_distribution = [
            (100, 0.2),    # Short sequences
            (500, 0.3),    # Medium sequences
            (1000, 0.3),   # Long sequences
            (5000, 0.15),  # Very long sequences
            (10000, 0.05)  # Extra long sequences
        ]
        
        for i in range(num_sequences):
            # Select length based on distribution
            rand = np.random.random()
            cumsum = 0
            length = 1000  # default
            
            for seq_length, prob in length_distribution:
                cumsum += prob
                if rand < cumsum:
                    length = seq_length
                    break
                    
            # Generate random sequence
            bases = ['A', 'T', 'C', 'G']
            sequence = ''.join(np.random.choice(bases, length))
            
            # Add some structure (optional)
            if np.random.random() < 0.3:
                # Insert a motif
                motif = 'GATAAG'  # GATA motif
                pos = np.random.randint(0, length - len(motif))
                sequence = sequence[:pos] + motif + sequence[pos + len(motif):]
                
            seq_obj = DNASequence(
                sequence=sequence,
                chromosome=f"chr{np.random.randint(1, 23)}",
                start=np.random.randint(1000000, 100000000),
                end=0  # Will be set based on length
            )
            seq_obj.end = seq_obj.start + length
            
            sequences.append(seq_obj)
            
        return sequences

    async def _benchmark_throughput(self, test_sequences: List[DNASequence]) -> Dict:
        """Benchmark throughput with different batch sizes"""
        results = {}
        
        for batch_size in [1, 10, 50, 100]:
            if batch_size > len(test_sequences):
                continue
                
            start_time = time.time()
            
            # Process in batches
            for i in range(0, min(100, len(test_sequences)), batch_size):
                batch = test_sequences[i:i + batch_size]
                await self.components['vllm'].analyze_sequences(
                    batch,
                    task='regulatory_detection'
                )
                
            elapsed = time.time() - start_time
            sequences_processed = min(100, len(test_sequences))
            
            results[f'batch_size_{batch_size}'] = {
                'sequences_per_second': sequences_processed / elapsed,
                'time_per_sequence': elapsed / sequences_processed
            }
            
        return results

    def _display_benchmark_results(self, results: Dict):
        """Display benchmark results"""
        self.console.print("\n[bold]Performance Benchmark Results[/bold]\n")
        
        # Regulatory detection time
        if 'regulatory_detection_time' in results:
            self.console.print(f"Regulatory detection (10 sequences): {results['regulatory_detection_time']:.2f}s")
            
        # Throughput results
        if 'throughput' in results:
            table = Table(title="Throughput Analysis")
            table.add_column("Batch Size", style="cyan")
            table.add_column("Sequences/Second", justify="right", style="green")
            table.add_column("Time/Sequence (ms)", justify="right", style="yellow")
            
            for batch_key, metrics in results['throughput'].items():
                batch_size = batch_key.split('_')[-1]
                table.add_row(
                    batch_size,
                    f"{metrics['sequences_per_second']:.1f}",
                    f"{metrics['time_per_sequence'] * 1000:.1f}"
                )
                
            self.console.print(table)

    def _save_pipeline_results(self, results: Dict, output_path: str):
        """Save custom pipeline results"""
        output_path = Path(output_path)
        
        # Convert any non-serializable objects
        serializable_results = {}
        for step_name, step_results in results.items():
            if isinstance(step_results, list):
                # Convert RegulatoryElement objects if present
                serializable = []
                for item in step_results:
                    if hasattr(item, '__dict__'):
                        serializable.append({
                            k: v for k, v in item.__dict__.items()
                            if not k.startswith('_')
                        })
                    else:
                        serializable.append(item)
                serializable_results[step_name] = serializable
            else:
                serializable_results[step_name] = step_results
                
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        self.console.print(f"[green]✓ Pipeline results saved to {output_path}[/green]")

    def _display_conservation_results(self, conservation_results: Dict):
        """Display cross-species conservation results"""
        # Ultra-conserved elements
        if conservation_results.get('ultra_conserved'):
            self.console.print("\n[bold]Ultra-Conserved Elements:[/bold]")
            
            table = Table()
            table.add_column("Type", style="cyan")
            table.add_column("Conservation", justify="right", style="green")
            table.add_column("Species", style="magenta")
            
            for uc_elem in conservation_results['ultra_conserved']:
                species_list = ', '.join(uc_elem['elements'].keys())
                table.add_row(
                    uc_elem['type'],
                    f"{uc_elem['conservation']:.1%}",
                    species_list
                )
                
            self.console.print(table)
            
        # Lineage-specific elements
        if conservation_results.get('lineage_specific'):
            self.console.print("\n[bold]Lineage-Specific Elements:[/bold]")
            
            for species, elements in conservation_results['lineage_specific'].items():
                if elements:
                    self.console.print(f"\n{species}: {len(elements)} unique elements")
                    for elem in elements[:3]:  # Show first 3
                        self.console.print(f"  - {elem.element_type} at position {elem.start}")
                        
        # Regulatory turnover
        if conservation_results.get('regulatory_turnover'):
            self.console.print("\n[bold]Regulatory Turnover:[/bold]")
            
            table = Table()
            table.add_column("Species Comparison", style="cyan")
            table.add_column("Shared Elements", justify="right")
            table.add_column("Total Elements", justify="right")
            table.add_column("Turnover Rate", justify="right", style="yellow")
            
            for comparison, data in conservation_results['regulatory_turnover'].items():
                table.add_row(
                    comparison.replace('_', ' → '),
                    str(data['shared']),
                    str(data['total']),
                    f"{data['turnover_rate']:.1%}"
                )
                
            self.console.print(table)

    def _save_cross_species_results(self, all_elements: Dict, 
                                  conservation_results: Dict, 
                                  output_path: str):
        """Save cross-species analysis results"""
        output_path = Path(output_path)
        
        results = {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'species_analyzed': list(all_elements.keys()),
            'elements_by_species': {},
            'conservation_analysis': conservation_results
        }
        
        # Convert elements to serializable format
        for species, elements in all_elements.items():
            results['elements_by_species'][species] = [
                {
                    'type': elem.element_type,
                    'chr': elem.chr,
                    'start': elem.start,
                    'end': elem.end,
                    'score': elem.score,
                    'length': elem.length
                }
                for elem in elements
            ]
            
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.console.print(f"[green]✓ Cross-species results saved to {output_path}[/green]")

    def _display_evolutionary_summary(self, results: Dict):
        """Display evolutionary analysis summary"""
        for gene_name, gene_data in results.items():
            tree = Tree(f"[bold]{gene_name}[/bold]")
            
            # Orthologs section
            ortho_branch = tree.add("Orthologs Found")
            for species, data in gene_data['orthologs'].items():
                species_info = ortho_branch.add(f"{species}")
                species_info.add(f"Enhancers: {data['enhancer_count']}")
                species_info.add(f"Total elements: {len(data['regulatory_elements'])}")
                
            # Evolution section
            if 'regulatory_evolution' in gene_data:
                evo_branch = tree.add("Evolutionary Patterns")
                
                evo_data = gene_data['regulatory_evolution']
                if 'ultra_conserved' in evo_data:
                    evo_branch.add(f"Ultra-conserved elements: {len(evo_data['ultra_conserved'])}")
                    
                if 'lineage_specific' in evo_data:
                    for species, elements in evo_data['lineage_specific'].items():
                        if elements:
                            evo_branch.add(f"{species}-specific: {len(elements)} elements")
                            
            self.console.print(Panel(tree, title=f"Gene: {gene_name}", border_style="blue"))

    async def _run_regulatory_step(self, step_config: Dict) -> Any:
        """Run regulatory detection step"""
        # Implementation depends on step configuration
        return []

    async def _run_expression_step(self, step_config: Dict) -> Any:
        """Run expression prediction step"""
        # Implementation depends on step configuration
        return {}

    async def _run_multimodal_step(self, step_config: Dict) -> Any:
        """Run multimodal analysis step"""
        # Implementation depends on step configuration
        return {}

# ======================== Main Entry Point ========================

def main():
    """Main entry point for BRIDGE orchestrator"""
    parser = argparse.ArgumentParser(
        description="BRIDGE Workflow Orchestrator - Central command for all analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sequence analysis
  python bridge_orchestrator.py single -s ATCGATCG... --analyze-genes
  
  # Batch analysis
  python bridge_orchestrator.py batch -i sequences.fasta -o results.json
  
  # Gene-centric analysis
  python bridge_orchestrator.py gene -g NANOG --upstream 1000000 --downstream 100000
  
  # Variant impact
  python bridge_orchestrator.py variant -v variants.vcf --detailed
  
  # Multi-modal analysis
  python bridge_orchestrator.py multimodal --dna DNA_SEQ --protein PROTEIN_SEQ
  
  # Cross-species analysis
  python bridge_orchestrator.py cross-species --reference-species human --target-species mouse rat
  
  # Custom pipeline
  python bridge_orchestrator.py custom -p my_pipeline.yaml
        """
    )
    
    # Global options
    parser.add_argument('--config', '-c', default='config/bridge_config.json',
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='workflow', help='Analysis workflow to run')
    
    # Single sequence analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single DNA sequence')
    single_parser.add_argument('-s', '--sequence', required=True,
                              help='DNA sequence to analyze')
    single_parser.add_argument('--chromosome', help='Chromosome name')
    single_parser.add_argument('--start', type=int, help='Start position')
    single_parser.add_argument('--end', type=int, help='End position')
    single_parser.add_argument('--analyze-genes', action='store_true',
                              help='Perform gene assignment and expression analysis')
    single_parser.add_argument('--cell-type', help='Cell type for expression prediction')
    
    # Batch analysis
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple sequences')
    batch_parser.add_argument('-i', '--input-file', required=True,
                             help='Input file (FASTA or text)')
    batch_parser.add_argument('-o', '--output', help='Output file')
    batch_parser.add_argument('--batch-size', type=int, default=100,
                             help='Batch size for processing')
    batch_parser.add_argument('--priority', type=int, default=5,
                             help='Processing priority (1-10)')
    
    # Genomic region
    region_parser = subparsers.add_parser('region', help='Analyze genomic region')
    region_parser.add_argument('--chromosome', required=True, help='Chromosome')
    region_parser.add_argument('--start', type=int, required=True, help='Start position')
    region_parser.add_argument('--end', type=int, required=True, help='End position')
    region_parser.add_argument('--genome', default='hg38', help='Reference genome')
    region_parser.add_argument('--visualize', action='store_true',
                              help='Generate visualization')
    region_parser.add_argument('-o', '--output', help='Output file for visualization')
    
    # Gene-centric
    gene_parser = subparsers.add_parser('gene', help='Gene-centric analysis')
    gene_parser.add_argument('-g', '--gene-name', required=True, help='Gene name')
    gene_parser.add_argument('--upstream', type=int, default=100000,
                            help='Upstream region size (bp)')
    gene_parser.add_argument('--downstream', type=int, default=10000,
                            help='Downstream region size (bp)')
    gene_parser.add_argument('--cell-type', help='Cell type for expression')
    
    # Variant impact
    variant_parser = subparsers.add_parser('variant', help='Variant impact analysis')
    variant_parser.add_argument('-v', '--variants-file', required=True,
                               help='Variants file (VCF or TSV)')
    variant_parser.add_argument('--detailed', action='store_true',
                               help='Show detailed results')
    variant_parser.add_argument('-o', '--output', help='Output file')
    
    # Expression analysis
    expr_parser = subparsers.add_parser('expression', help='Expression prediction')
    expr_parser.add_argument('-g', '--genes-file', required=True,
                            help='Gene list file')
    expr_parser.add_argument('--cell-type', default='generic',
                            help='Cell type')
    expr_parser.add_argument('--heatmap', action='store_true',
                            help='Generate expression heatmap')
    expr_parser.add_argument('-o', '--output', help='Output file')
    
    # Multi-modal
    multi_parser = subparsers.add_parser('multimodal', help='Multi-modal analysis')
    multi_parser.add_argument('--dna', '--dna-sequence', dest='dna_sequence',
                             help='DNA sequence')
    multi_parser.add_argument('--rna', '--rna-sequence', dest='rna_sequence',
                             help='RNA sequence')
    multi_parser.add_argument('--protein', '--protein-sequence', dest='protein_sequence',
                             help='Protein sequence')
    
    # Conservation
    cons_parser = subparsers.add_parser('conservation', help='Conservation analysis')
    cons_parser.add_argument('--reference-species', default='human',
                            help='Reference species')
    cons_parser.add_argument('--target-species', nargs='+',
                            default=['mouse', 'rat', 'zebrafish'],
                            help='Target species for comparison')
    
    # Regulatory design
    design_parser = subparsers.add_parser('design', help='Design regulatory elements')
    design_parser.add_argument('--target-genes', nargs='+', required=True,
                              help='Target genes')
    design_parser.add_argument('--expression-level', type=float, default=2.0,
                              help='Desired expression fold-change')
    
    # Network analysis
    network_parser = subparsers.add_parser('network', help='Regulatory network analysis')
    network_parser.add_argument('-g', '--genes-file', required=True,
                               help='Gene list file')
    network_parser.add_argument('--export-format', choices=['json', 'graphml', 'cytoscape'],
                               help='Network export format')
    network_parser.add_argument('-o', '--output', help='Output file')
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Performance benchmark')
    bench_parser.add_argument('--num-sequences', type=int, default=100,
                             help='Number of test sequences')
    
    # Custom pipeline
    custom_parser = subparsers.add_parser('custom', help='Run custom pipeline')
    custom_parser.add_argument('-p', '--pipeline-config', required=True,
                              help='Pipeline configuration file (YAML)')
    custom_parser.add_argument('-o', '--output', help='Output file')
    
    # Cross-species analysis
    cross_parser = subparsers.add_parser('cross-species', help='Cross-species regulatory analysis')
    cross_parser.add_argument('--reference-species', default='human',
                             help='Reference species')
    cross_parser.add_argument('--target-species', nargs='+', required=True,
                             help='Target species to compare')
    cross_parser.add_argument('--input-format', choices=['coordinates', 'file'],
                             default='coordinates', help='Input format')
    cross_parser.add_argument('--chromosome', help='Chromosome (for coordinates)')
    cross_parser.add_argument('--start', type=int, help='Start position')
    cross_parser.add_argument('--end', type=int, help='End position')
    cross_parser.add_argument('--input-file', help='Input file (for file format)')
    cross_parser.add_argument('-o', '--output', help='Output file')
    
    # Evolutionary analysis
    evo_parser = subparsers.add_parser('evolution', help='Evolutionary regulatory analysis')
    evo_parser.add_argument('--gene-list', nargs='+', required=True,
                           help='Genes to analyze')
    evo_parser.add_argument('--species-list', nargs='+', required=True,
                           help='Species to compare')
    evo_parser.add_argument('--upstream', type=int, default=100000,
                           help='Upstream region size')
    evo_parser.add_argument('--downstream', type=int, default=10000,
                           help='Downstream region size')
    evo_parser.add_argument('-o', '--output', help='Output file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Disable GPU if requested
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    # Check if workflow specified
    if not args.workflow:
        parser.print_help()
        sys.exit(1)
        
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(args.config)
    
    # Run workflow
    try:
        # Initialize components
        console.print("[bold]Initializing BRIDGE components...[/bold]")
        asyncio.run(orchestrator.initialize())
        
        # Execute selected workflow
        console.print(f"\n[bold]Running {args.workflow} workflow...[/bold]\n")
        workflow_func = orchestrator.workflows[args.workflow]
        asyncio.run(workflow_func(args))
        
        console.print("\n[green]✓ Analysis complete![/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        if args.log_level == 'DEBUG':
            console.print_exception()
        sys.exit(1)
    finally:
        # Cleanup
        if hasattr(orchestrator, 'components') and 'vllm' in orchestrator.components:
            orchestrator.components['vllm'].shutdown()

if __name__ == "__main__":
    main()