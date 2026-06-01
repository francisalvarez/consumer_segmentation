# Consumer Segmentation Workflow

This project contains a reusable Python workflow for consumer segmentation analysis using the `Mall_Customers`{https://www.kaggle.com/datasets/shrutimechlearn/customer-data/data} dataset. It is designed for reproducibility, interpretability, statistical rigor, and business actionability.

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

# Review of using agents in VScode. 
My first impression is majority of the boilerplate code was produced efficiently and accurately. I believe there is still need for an analyst-in-the-loop to QC the results and review the decisions.

## Number of clusters.
The initial number of clusters, $k$, was chosen based on the number that of clusters with the largest silhoutte score. From the PCA figures, classifying the data with five groups, instead of six , seems like a reasonable option based on visual inspection. Also, reducing the number of groups simplifies the analysis so the results are more memorable for the audience.

Silhoutte Score Note:<br>
Near \(+1\): Points are well-matched to their own cluster and far away from neighboring clusters. The clustering configuration is strong.Near \(0\): Points are very close to the decision boundary between two neighboring clusters, indicating overlapping clusters.Near \(-1\): Points may have been assigned to the wrong cluster, as they are likely closer to a different cluster than the one they were placed in.

## Perceptual Map
I chose to add a perceptual map to the PCA figure.

### Perceptual Map Notes
**What does the arrow length indicate?**
In a PCA perceptual map (often called a biplot), the length of a variable arrow indicates how strongly that variable contributes to the two PCA dimensions being displayed.

Long arrows mean:

The variable is well represented by PCA1 and PCA2.
The variable explains a lot of the variation seen in the map.
Differences between clusters are strongly associated with that variable.

Short arrows mean:

The variable is not well captured by these two dimensions.
Some of its variation exists in later principal components (PC3, PC4, etc.).
It is less useful for interpreting cluster separation on this specific map.

In your plot:

Annual Income has the longest arrow → income is one of the strongest drivers of segment separation.
Spending Score is also quite long → spending behavior strongly differentiates customers.
Age is much shorter → age contributes less to the separation visible in the first two principal components.