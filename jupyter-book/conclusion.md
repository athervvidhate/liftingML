# Final Notes
* Instead of the current method of training a regular RoBERTa model on the lifting corpus and then wrapping it to work like a sentence transformer, we could have used Text-Search Data Augmentation for Embeddings (TSDAE) to train the sentence embeddings directly. Theoretically, this would have better performance in encoding the semantic similarity.
    * The reason that method wasn't done here was because I already fine-tuned a token-level roBERTa model on this data, and it still worked much better than the base sentence-transfomers `all-miniLM-L6-V2` model.
    * I also wanted to gain more experience working with PyTorch, and the method used allowed me to get my hands dirty.

* A limitation with this model is that it won't be able to reliably find programs that have a very low intensity rating, as those types of programs rarely show up in the dataset. 
    * The data was sourced from user generated programs from [Boostcamp](https://www.boostcamp.app/), a workout tracking app. People that download this app are much more likely to be passionate about fitness, and low intensity workouts (generally) don't result in as much gains as medium to high intensity workouts. The subset of people with this app will be the ones who strive for higher gains. 
    * Low intensity workouts are great for people just getting into fitness or people who just want to be a little healthier. 
    * Some future steps to improve this recommendation system would be to source more of this low intensity workouts across a variety of use cases to increase its generalizability.

* The fine-tuned roBERTa model is too large to run on free-tier cloud services (GCP, Streamlit Community Cloud), so just for cloud deployment I trained a smaller ALBERT model as well. It won't provide the best possible recommendations, but it should still be extremely serviceable.

* If you'd like to get the best possible recommendations, clone this repo and run the Streamlit app locally.

## Credits
Dataset from Adnane Louardi: [Link](https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset/code)