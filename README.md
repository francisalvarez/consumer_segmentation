# Consumer Segmentation Workflow

This project contains a reusable Python workflow for consumer segmentation analysis using the `Mall_Customers` dataset. It is designed for reproducibility, interpretability, statistical rigor, and business actionability.

## Structure

- `data/`: local dataset storage
- `src/`: reusable segmentation modules
- `outputs/`: saved reports and processed outputs
- `figures/`: saved charts and visualizations
- `notebooks/`: analysis notebooks and narrative summaries
- `main.py`: pipeline entrypoint for data loading, preprocessing, clustering, and profiling

## Usage

Activate the existing `data_processing` environment and run:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data_processing
python main.py
```

## Notes

- The workflow is built to support additional consumer datasets and can be extended with more clustering methods or domain-specific features.
- The current dataset is the Kaggle `shrutimechlearn/customer-data` Mall Customers dataset.
