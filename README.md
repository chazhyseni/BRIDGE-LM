# BRIDGE-LM

<div align="center">

## Biological Regulatory Integration & Detection through Genomic Language Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

*A hierarchical transformer-based framework that orchestrates multiple genomic language models to detect regulatory elements with both genome-wide context and nucleotide-level precision*

[**📚 Full Documentation**](https://chazhyseni.github.io/BRIDGE-LM/) | [**🚀 Quick Start**](#quick-start) | [**📊 Examples**](#examples)

</div>

---

## 🎯 The Challenge

Regulatory elements (enhancers and promoters) pose unique detection challenges:

- **Distance**: Can be up to 1 million base pairs away from target genes
- **Size variability**: Enhancers range from 100-2000bp with no fixed length  
- **Fuzzy boundaries**: Unlike genes, regulatory elements have gradual transitions
- **Context-dependent**: Same sequence can be active in one cell type but not another
- **Evolutionary flexibility**: Regulatory sequences evolve rapidly

Traditional methods miss complex patterns. Single deep learning models face a fundamental tradeoff between long-range context and nucleotide precision.

## 💡 The BRIDGE Solution

**Key Insight**: Instead of building one model that does everything adequately, BRIDGE orchestrates multiple specialized models, each excelling at what it does best, in a biologically-informed pipeline.

### Three-Stage Hierarchical Approach

```
Input DNA (up to 450kb)
        ↓
┌─────────────────────────────────────┐
│ Stage 1: Long-Range Scanning        │
│ HyenaDNA (450kb context)            │
│ → Coarse predictions                │
│ → Candidate regions                 │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 2: Precise Boundary Detection │
│ DNABERT-2 (512bp windows)           │
│ → Token-level analysis              │
│ → Attention gradients               │
│ → Exact boundaries                  │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 3: Multi-Signal Integration   │
│ • LucaOne validation                │
│ • DNA shape features                │
│ • Conservation scores               │
│ • Motif analysis                    │
└─────────────────────────────────────┘
        ↓
Validated Regulatory Elements
```

## 🌟 Key Features

- 🎯 **Multi-scale analysis**: Combines long-range (450kb) and short-range (512bp) models
- 🌍 **Cross-species support**: Works across human, mouse, zebrafish, and other species using mean pooling
- ⚡ **Production-ready**: vLLM optimization, REST API, batch processing, and continuous serving
- 🧬 **Multi-modal validation**: DNA, RNA, and protein sequence integration
- 📊 **Expression prediction**: Links regulatory elements to gene expression
- 🔧 **Modular design**: Easy to extend with new models or analysis types
- 🎨 **Precise boundaries**: Multi-signal integration for accurate element delineation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/chazhyseni/BRIDGE-LM.git
cd BRIDGE-LM

# Create conda environment
conda env create -f environment.yml
conda activate dna-models

# Install BRIDGE
pip install -e .

# Download pre-trained models (50GB)
python scripts/download_models.py
python scripts/download_data.py --genomes hg38 mm10
```

### Basic Usage

```python
from bridge import WorkflowOrchestrator
import asyncio

# Initialize BRIDGE
orchestrator = WorkflowOrchestrator()
await orchestrator.initialize()

# Analyze a sequence
from bridge.core.models import DNASequence
seq = DNASequence(
    sequence="ATCGATCG" * 1000,  # 8kb sequence
    chromosome="chr12",
    start=7940000,
    end=7948000
)

# Get results
results = await orchestrator.components['model_manager'].predict_regulatory_elements(seq)

# Print detected elements
for element in results:
    print(f"{element.element_type}: {element.chr}:{element.start}-{element.end}")
    print(f"  Score: {element.score:.3f}")
    print(f"  Core region: {element.core_start}-{element.core_end}")
    print(f"  TF motifs: {', '.join([m['tf'] for m in element.tf_motifs[:3]])}")
```

### Command Line Interface

```bash
# Analyze single sequence
bridge analyze -s ATCGATCG... --output results.json

# Batch analysis
bridge batch -i sequences.fasta -o results.bed --format bed

# Gene-centric analysis
bridge gene -g NANOG --upstream 1000000 --downstream 100000

# Start API server
bridge serve --host 0.0.0.0 --port 8000
```

## 📊 Examples

### Cross-Species Analysis

```python
# Compare regulatory landscapes across species
results = await orchestrator.cross_species_analysis(
    reference_species="human",
    target_species=["mouse", "zebrafish"],
    chromosome="chr2",
    start=176000000,
    end=176100000
)

# Find ultra-conserved elements
for element in results['ultra_conserved']:
    print(f"Conserved {element['type']}: {element['conservation']:.1%}")
```

### Multi-Modal Validation

```python
# Validate DNA→RNA→Protein relationships
validation = await orchestrator.multimodal_analysis(
    dna_sequence="ATCG...",
    rna_sequence="AUGC...",
    protein_sequence="MKVL..."
)

print(f"Central dogma valid: {validation['central_dogma']['all_valid']}")
print(f"Protein-DNA binding score: {validation['protein_dna_interaction']['score']}")
```

## 🔬 Technical Innovations

### 1. Attention Gradient Boundary Detection
DNABERT-2's attention patterns show sharp transitions at regulatory boundaries. We compute attention gradients and use peaks to identify precise start/end positions.

### 2. Multi-Signal Integration
Different signals are weighted based on element type:
- **Enhancers**: Higher weight on motif density and nucleosome depletion
- **Promoters**: Higher weight on attention gradients and conservation

### 3. Cross-Species Mean Pooling
HyenaDNA's mean pooling creates species-agnostic representations, enabling analysis of any species without retraining.

## 💻 System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (V100/A100 recommended)
- **RAM**: 32GB minimum
- **Storage**: 50GB for models and reference data
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.9-3.10

## 📈 Performance

- **Throughput**: 1000+ sequences/second (with GPU)
- **Memory efficiency**: Handles 450kb sequences
- **Accuracy**: Validated on VISTA enhancers and ENCODE cCREs
- **Scalability**: Distributed processing support

## 🛠️ Advanced Features

- **Fine-tuning**: LoRA-based adaptation for cell types
- **Custom pipelines**: YAML-based workflow definition
- **Variant impact**: Predict effects of mutations
- **CRISPR design**: Target regulatory cores
- **Expression prediction**: Link elements to genes

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

BRIDGE-LM builds upon excellent work by:
- [HyenaDNA](https://github.com/HazyResearch/hyena-dna) 
- [DNABERT-2](https://github.com/Zhihan1996/DNABERT_2) 
- [LucaOne](https://github.com/LucaOne/LucaOne) 

## 📧 Contact

- **Email**: chaz.hyseni@gmail.com
- **Issues**: [GitHub Issues](https://github.com/chazhyseni/BRIDGE-LM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chazhyseni/BRIDGE-LM/discussions)

---

<div align="center">
<b>BRIDGE-LM</b> - Advancing genomic understanding through hierarchical model orchestration
</div>
