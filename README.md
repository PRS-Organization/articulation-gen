# Articulation-Gen: 3D Part Segmentation and Articulated Object Generation (coming soon!)

[![Hugging Face](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/your-model-repo) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
Articulation-Gen integrates 3D semantic segmentation, physics-guided joint optimization, and LLM-augmented URDF synthesis, complemented by a large-scale hinge asset dataset, to generate physically compliant multi-joint 3D objects.
**Articulation-Gen** (coming soon!) is an open-source framework for 3D part segmentation and articulated object generation, featuring:  
- A complete 3D asset generation pipeline  
- Deep learning-based semantic segmentation  
- Automatic joint generation for articulated objects

![Example Output](docs/demo.png)

---

## Key Features
- ✅ **Part Segmentation** - Semantic segmentation using PointNet++  
- ✅ **Joint Generation** - Automatic physical constraint creation  
- ✅ **Multi-Format Support** - Supports OBJ/GLB/FBX/USDZ formats  
- ✅ **HuggingFace Integration** - Pretrained model loading

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/Articulation-Gen.git
cd Articulation-Gen
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt

```
### 3. Download the dataset
### 4. Run the Generation Pipeline
```bash
python main.py --input input.obj --output output.glb --task full_pipeline
```
Submit issues via GitHub Issues