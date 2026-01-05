# DOCM to Nifti Preprocessing Mix

Welcome to the **DOCM to Nifti Preprocessing Mix** repository! This project provides tools and scripts to preprocess and convert medical imaging data formats, including **DOCM** (DICOM) files, into the **Nifti** format, with additional preprocessing options. The primary language used in the repository is Python.

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About the Project

Medical imaging often involves working with large-scale medical datasets in formats like **DICOM** (Digital Imaging and Communications in Medicine). This repository simplifies the conversion and preprocessing pipeline for transforming **DICOM** files to **Nifti** format, which is more commonly used in analysis tools for neuroimaging and medical data processing.

## Features

- **DICOM to Nifti Conversion**: Transform DICOM files into Nifti format.
- **Preprocessing Options**: Includes features like normalization, resampling, and intensity correction.
- **100% Python Implementation**: Easy to manage and extend.

## Getting Started

To get started with the repository, follow the steps below:

### Prerequisites

Ensure you have the following tools installed:
- Python 3.x
- pip (Python package manager)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zqrxiangxueIt/docm_to_nfiti_preprocessing_mix
   cd docm_to_nfiti_preprocessing_mix
   ```

## Usage

Run the preprocessing script:

```bash
python preprocess.py --input [input_folder] --output [output_folder] --options [options]
```

Where:
- `input`: The folder containing DICOM files.
- `output`: The folder to store the resulting Nifti files.
- `options`: Optional preprocessing parameters (e.g., --normalize, --resample).

For detailed usage and examples, check the `docs/` folder.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Clone it to your local environment.
3. Create a new branch for your feature (`git checkout -b feature-name`).
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push your branch (`git push origin feature-name`).
6. Create a pull request.

Ensure your code adheres to the PEP 8 guidelines.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

---

We hope this repository aids your medical imaging preprocessing tasks! 🌟 If you encounter any bugs or have feature requests, please create an issue or reach out.
