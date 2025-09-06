# Low-Field MRI Framework

A deep learning-based image enhancement pipeline for improving low-field MRI scans to enable early, non-hospitalizing detection of Alzheimer's Disease (AD).

## Overview

This framework implements a Super-Resolution Convolutional Neural Network (SRCNN) that enhances the quality of low-field (LF) MRI scans to approximate high-field (HF) MRI quality, enabling better downstream segmentation and diagnosis.

## Features

- **SRCNN-based Enhancement**: Deep learning model for MRI super-resolution
- **Alzheimer's Detection**: Specialized pipeline for AD detection workflows
- **Quality Improvement**: Enhances low-field MRI scans to high-field quality standards

## Installation

```bash
git clone https://github.com/Reteena/Low-Field-MRI-Framework-Open-Source.git
cd Low-Field-MRI-Framework-Open-Source
pip install -r requirements.txt
```

## Usage

```python
# Basic usage example
from lf_mri_framework import LowFieldEnhancer

enhancer = LowFieldEnhancer()
enhanced_image = enhancer.enhance('path/to/lowfield_scan.nii.gz')
```

## Pipeline

```
Low-Field MRI → SRCNN Enhancement → Enhanced MRI → Segmentation → AD Detection
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. Please check the repository for license details.

## Citation

If you use this framework in your research, please cite this repository.

---

Built by Reteena
