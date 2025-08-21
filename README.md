# liftingML — ML-Powered Workout Program Recommender

This project is an end-to-end machine learning application that recommends complete workout programs based on a user’s preferences and goals. It combines domain-tuned sentence embeddings with structured program features to retrieve similar, high-quality routines from a large corpus of programs. The application is delivered as an interactive Streamlit web app and can be deployed to Google App Engine (Flexible) with models and data hosted in Google Cloud Storage.

## What it does

- Accepts a free-form description of what the user wants (goals, style, constraints) along with numeric and categorical preferences (intensity, program length, time per workout, equipment, experience level, and more).
- Encodes the text using a transformer-based sentence embedder and fuses it with structured features representing a workout program.
- Computes similarity against a database of existing programs to return the most relevant, complete plans, each with metadata and the full week-by-week, exercise-by-exercise breakdown.
- Offers two model options: a faster ALBERT variant and a higher-accuracy RoBERTa variant.

## Data and features

The recommender draws from a large dataset of workout programs with both metadata and detailed schedules. Three primary artifacts are used at runtime:

- cleaned_600k.csv: a program-level dataset with detailed exercise breakdowns used to display full plans for recommendations.
- program_features.csv: a compact representation of program-level metadata (titles, descriptions, attributes).
- final_features.csv: a feature matrix used for clustering and similarity search. It contains engineered numeric features (e.g., average reps per exercise, total sets and reps per week, program length, time per workout) and one-hot encodings of categorical attributes (e.g., level, goal, equipment). The text description embedding is concatenated to these numeric features to create a single joint representation per program.

At query time, the app constructs the same joint representation for the user’s input by:

1) One-hot encoding level, goal, and equipment selections.
2) Combining user-provided numeric preferences (e.g., intensity, program length, time per workout, sets/reps) with the text embedding of the free-form description.
3) Standardizing numeric features to match the training distribution of the feature matrix.

## Modeling approach

The system uses a transformer-based sentence encoder to represent free-text program descriptions:

- ALBERT-based embedder: optimized for lower latency and smaller footprint.
- RoBERTa-based embedder: optimized for higher-quality embeddings.

Both variants are wrapped by a `CustomSentenceEmbedder` that:

- Loads model weights and tokenizers either from Google Cloud Storage (gs:// buckets) or from local directories.
- Performs mean pooling over the last hidden states with attention masking and optional L2 normalization to produce a sentence-level vector.
- Exposes a simple `encode()` interface for batch inference on CPU.

## Recommendation logic

- K-Means clustering (default k=25) groups programs in the feature space. This supports an optional “within cluster only” retrieval mode for more focused recommendations.
- Cosine similarity between the user’s joint vector and the program feature matrix drives nearest-neighbor retrieval.
- Top-N results are returned with:
  - Program metadata (title, description, level, goal, equipment, length, time per workout, number of exercises)
  - The full exercise list (week, day, exercise name, sets, reps, intensity)
- Users can download any recommended program as a CSV.

## Application experience

On the first run after startup, the app caches models, data, and clustering artifacts to minimize repeated latency. Subsequent queries are fast and incremental.

## Infrastructure and deployment

- The app is containerized with a slim Python 3.12 base image. It installs a CPU-only PyTorch wheel and project dependencies. The container exposes port 8080 for Streamlit.
- Google App Engine (Flexible) is used for deployment, with a custom runtime configuration and autoscaling between 1–2 instances by default. CPU and memory allocations are specified to balance responsiveness with cost.
- Model and data artifacts are fetched from Google Cloud Storage using `gcsfs`. If cloud access fails, the application falls back to local copies baked into the image or present on disk.
- Environment variables:
  - MODEL_BUCKET: the GCS bucket containing model directories (e.g., `albert_finetuned`, `roberta_finetuned`).
  - DATA_BUCKET: the GCS bucket containing CSV artifacts (`cleaned_600k.csv`, `program_features.csv`, `final_features.csv`).

## Design choices and trade-offs

- Hybrid representation: Combining semantic text embeddings with structured numeric features captures both program intent and concrete workload characteristics. This yields better retrieval quality than either modality alone.
- Dual-model option: ALBERT is provided for lower-latency use cases and smaller instance types; RoBERTa is available for users who prioritize quality over speed.
- Clustering for control: K-Means provides an optional constraint to keep recommendations in the same neighborhood, which can improve relevance when users want variations on a theme rather than broad exploration.
- CPU-first inference: The default deployment targets CPU for cost and portability. This is sufficient for interactive use with caching; GPU support can be added down the line for heavier workloads.

## Additional resources

- Project walkthrough and methodology overview: https://workout-walkthrough.atherv.com