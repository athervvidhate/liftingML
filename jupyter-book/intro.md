# Workout Program Recommender

Hello! Welcome to the walkthrough of my Workout Program Recommendation System. This project was created to enhance my understanding of various classical machine learning and deep learning techniques. We will delve into ML techniques such as KMeans, PCA, BERT, and fine-tuning via PyTorch. This site is divided into each step of the process I took in this project. Get ready to dive in!

## Project Breakdown
1. **Data Cleaning and Exploratory Data Analysis**
    * Examine the raw workout program data, clean inconsistencies, handle missing values, and perform EDA to uncover patterns and insights that will guide the rest of the project.

2. **Feature Engineering**
    * Develop and extract meaningful features from the cleaned dataset to better capture the unique characteristics of each workout program.

3. **Initial Model Training and Recommendation**
    * Create initial model using a Sentence Transformer for description embeddings, KMeans for clustering, and PCA for visualization. Use cosine similarity to find similar workouts to one another. Find out the issues with using a pretrained model for this task.

4. **Fine-tuning Model and Updated Recommendation**
    * Fine tune a roBERTa model on the user-generated text of the dataset, then wrap the model in a custom made SentenceEmbedder class to create sentence level embeddings. Recreate sentence embeddings using the fine tuned model, and remake feature dataset using the initial process. Utilize KMeans for clustering again and view example program recommendations.

5. **Final Notes**
    * Briefly contrast other potential methods with the methods used in this projects and touch on limitations of the model from the input dataset. Explain the use of a smaller, less intelligent model in the deployed app and explain how to use full model locally.