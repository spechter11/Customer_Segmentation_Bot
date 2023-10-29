# CustomerSegmentationBot
Automated customer segmentation tailored for clothing retail using machine learning.

## Description

This bot is designed to automate the process of segmenting customers in the clothing retail industry. Using machine learning algorithms and a synthetic dataset, it clusters customers based on various features like age, location, and purchase history.

## Features

- **Data Validation**: Ensures all required columns are present in the dataset.
- **Tailored for Clothing Retail**: Specifically designed to handle datasets relevant to the clothing industry.
- **Configurable**: Allows the user to choose the clustering algorithm and the number of clusters.
- **Error Handling**: Comprehensive error handling mechanisms for robustness.
- **Export Features**: Capabilities to export clustering summaries to CSV and Word documents.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/spechter11/CustomerSegmentationBot.git
    ```
   
2. Navigate into the project directory:
    ```bash
    cd CustomerSegmentationBot
    ```
   
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Using Configuration File

1. Edit the `config.json` to specify parameters like `file_path`, `clustering_algorithm`, and `n_clusters`.

2. Run the bot:
    ```bash
    python main.py
    ```

## Contribution

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details
