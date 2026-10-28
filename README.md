# Netflix Movies and TV Shows Clustering

An end-to-end data science project that explores Netflix catalogue metadata and groups similar movies and TV shows using Natural Language Processing (NLP) and unsupervised machine learning.

## Project Overview

Netflix offers a diverse catalogue of movies and television shows across genres, countries, release years, and content categories. Understanding this catalogue can help analysts explore content patterns and investigate ways to organize similar titles.

This project uses exploratory data analysis, statistical hypothesis testing, text preprocessing, TF-IDF feature extraction, dimensionality reduction, and clustering algorithms to analyze Netflix titles.

It also explores a basic content-based recommendation approach using textual similarity.

## Objectives

- Explore and understand Netflix catalogue metadata.
- Clean the dataset and handle missing or inconsistent values.
- Analyze content types, genres, countries, release years, ratings, and other available attributes.
- Create visualizations to identify patterns in the catalogue.
- Perform statistical hypothesis testing.
- Transform textual metadata into numerical features using TF-IDF.
- Apply dimensionality reduction using Truncated SVD.
- Compare multiple unsupervised clustering algorithms.
- Evaluate cluster quality using silhouette scores and cluster inspection.
- Explore content-based recommendations using cosine similarity.

## Technologies Used

| Category | Tools and Libraries |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Natural Language Processing | NLTK |
| Feature extraction | Scikit-learn TF-IDF Vectorizer |
| Dimensionality reduction | Truncated SVD |
| Machine learning | K-Means, MiniBatch K-Means, Agglomerative Clustering |
| Evaluation | Silhouette Score, cluster-size analysis |
| Model persistence | Joblib |
| Development environment | Google Colab |

## Project Workflow

### 1. Data Loading and Exploration
- Load the Netflix dataset into a Pandas DataFrame.
- Inspect dataset dimensions, column names, data types, missing values, and duplicate records.
- Understand the available metadata.

### 2. Data Cleaning and Exploratory Data Analysis
- Prepare relevant columns for analysis.
- Investigate missing values and duplicate records.
- Explore content types, genres, countries, release years, ratings, and durations.
- Visualize patterns using appropriate charts.

### 3. Statistical Hypothesis Testing
Perform statistical tests to investigate selected relationships in the dataset, such as differences in release-year distributions and associations between content type and ratings.

Interpret p-values using a defined significance level and consider the assumptions of each test.

### 4. Text Preprocessing
Prepare relevant textual metadata through operations such as:
- Lowercasing and text normalization.
- Removing punctuation and unwanted characters.
- Stopword removal.
- Tokenization and stemming where appropriate.
- Combining relevant metadata into a text representation.

### 5. Feature Engineering
- Convert text into numerical features using TF-IDF.
- Reduce dimensionality using Truncated SVD where appropriate.
- Prepare feature representations for clustering and evaluation.

### 6. Clustering Models
The project explores three clustering approaches:

- **K-Means:** Groups titles into clusters by minimizing within-cluster squared distances.
- **MiniBatch K-Means:** Provides a more computationally efficient alternative for larger datasets.
- **Agglomerative Clustering:** Builds hierarchical groups by progressively merging observations.

The notebook compares candidate cluster counts and linkage settings using a reproducible sample where appropriate.

### 7. Model Evaluation
- Use silhouette scores to assess cluster separation and cohesion.
- Examine cluster-size distributions.
- Inspect representative titles and cluster terms.
- Consider computational cost and interpretability when comparing models.

Since this is an unsupervised learning task, there are no predefined ground-truth labels for ordinary classification accuracy.

### 8. Content-Based Recommendation
Explore a simple recommendation function that uses cosine similarity between textual feature vectors to identify titles with similar metadata.

### 9. Model Persistence
Save and reload selected fitted components using Joblib and perform a sanity check on new text examples.

## Dataset

The project uses a Netflix Movies and TV Shows catalogue dataset containing metadata such as:

- Title and content type.
- Director and cast.
- Country.
- Date added and release year.
- Rating and duration.
- Listed genres or categories.
- Description.

**Dataset setup:** The notebook expects the CSV file to be available in Google Drive. Update the file path in the dataset-loading cell to match your own copy.

The dataset is not included in this repository. Use a dataset source you are permitted to access and redistribute, and follow its applicable terms.

## How to Run the Project

### Option 1: Google Colab

1. Open the notebook from this repository in Google Colab.
2. Make a copy in your Google Drive.
3. Obtain the dataset and place it in your Drive or upload it when prompted by the notebook.
4. Update the CSV file path if necessary.
5. Run the cells from top to bottom.
6. Review the visualizations, hypothesis-testing results, clustering metrics, and recommendations.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/netflix-movies-tv-shows-clustering.git
cd netflix-movies-tv-shows-clustering
```

Create and activate a virtual environment if desired, then install the required packages:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn scipy wordcloud joblib
```

Launch Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

Open the project notebook and update the dataset path for your local environment.

> Note: The notebook was designed for Google Colab. Local execution may require adjustments to Google Drive mounting, dataset paths, and NLTK resource downloads.

## Evaluation Approach

The primary clustering metric is the silhouette score.

- Scores closer to **1** generally indicate better-separated clusters.
- Scores around **0** suggest overlapping clusters.
- Negative scores can indicate that some observations may fit another cluster better.

Cluster quality should also be assessed by reviewing actual title groupings. A high silhouette score alone does not guarantee meaningful recommendations or improved user engagement.

Add your final model comparison table and verified numerical results here after running the notebook successfully.

## Key Learning Outcomes

- Practical data cleaning and exploratory data analysis.
- Statistical hypothesis testing.
- NLP preprocessing and TF-IDF feature engineering.
- Dimensionality reduction for high-dimensional text data.
- Implementation and comparison of unsupervised clustering algorithms.
- Model evaluation and persistence.
- Building a basic content-similarity recommendation prototype.

## Limitations and Future Improvements

- TF-IDF captures word-based similarity but does not fully understand semantic meaning.
- Catalogue metadata may contain missing or incomplete information.
- Clusters may not align perfectly with human-defined genres or individual user preferences.
- The prototype does not use actual viewing history or user engagement data.

Potential improvements include semantic embeddings, more advanced recommendation methods, improved cluster interpretation, and a web application for interactive recommendations.

## Author

**Mohammed Shoaib**

BCA Graduate | Aspiring Data Analyst

Skills demonstrated: Python, Pandas, Data Visualization, SQL-related analytical thinking, NLP, and Unsupervised Machine Learning.

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://www.linkedin.com/in/shoaib9/

---

*This project was developed for educational and portfolio purposes. Netflix is a trademark of its respective owner. This project is not affiliated with or endorsed by Netflix.*
