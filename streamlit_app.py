import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import ast

# Add parent directory to path to import RobertaSentenceEmbedder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CustomSentenceEmbedder import CustomSentenceEmbedder

# Set page config with responsive sidebar
st.set_page_config(
    layout="wide", 
    page_title='ML Workout Program Recommender',
    page_icon=None,
    initial_sidebar_state="collapsed"  # Default to collapsed for better mobile experience
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    /* Responsive sidebar behavior */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
    }
    
    /* Main styling - adapts to theme */
    .main {
        background: var(--background-color);
        padding: 0;
    }
    
    /* Header styling - adapts to theme */
    .main-header {
        background: var(--background-color);
        padding: 2rem 0;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }
    
    .main-header h1 {
        color: var(--text-color);
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
    }
    
    .main-header p {
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Form styling - adapts to theme */
    .stForm {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }
    
    /* Input styling - adapts to theme */
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect, .stSlider {
        background: var(--background-color);
        border-radius: 12px;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus, .stMultiselect:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Results styling - adapts to theme */
    .results-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin-top: 2rem;
        border: 1px solid var(--border-color);
    }
    
    .program-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .program-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: #667eea;
    }
    
    .program-title {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .program-meta {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .meta-label {
        font-weight: 600;
        color: var(--text-color);
        min-width: 120px;
    }
    
    .meta-value {
        color: var(--text-color);
        margin-left: 0.5rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: var(--background-color);
        color: var(--text-color);
        border: 2px solid var(--border-color);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .stForm {
            padding: 1rem;
        }
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    

</style>
""", unsafe_allow_html=True)

# Initialize the embedder
@st.cache_resource
def load_embedder():
    # Change folder to roberta_finetuned for the best performance
    # Change device to 'mps' if you're on an apple silicon device
    return CustomSentenceEmbedder.load('./albert_finetuned', device='cuda' if torch.cuda.is_available() else 'cpu')

embedder = load_embedder()

@st.cache_data
def load_data(url_huge_data, url_program_features, url_final_features):
    huge_data = pd.read_csv(url_huge_data)
    program_features = pd.read_csv(url_program_features)
    final_features = pd.read_csv(url_final_features).drop(columns=['Unnamed: 0'])
    return huge_data, program_features, final_features

huge_data, program_features, final_features = load_data(
    './data/cleaned_600k.csv',
    './data/program_features.csv',
    './data/final_features_albert.csv'
)

def find_top_n(similarity_matrix, n_programs, program, metadata, info, cluster=None, features=None):
    """
    Gets the top n workout programs.

    Args:
        similarity_matrix (np.ndarray): Matrix of similarity scores between programs.
        n_programs (int): Number of top similar programs to return.
        program (int): Index of the program to compare against.
        metadata (list): List of metadata column names to include in the result.
        info (list): List of info column names to include in the result.

    Returns:
        list[pd.DataFrame]: List of DataFrames, each containing the metadata and info for a top similar program.
    """
    scores = similarity_matrix[program]

    if cluster:
        mask = (features['cluster'] == cluster).values
        scores = scores * mask

    idxs = np.argsort(scores)[::-1]

    # Gets the top n indices that aren't itself
    top_n = idxs[idxs != program][:n_programs]
    top_titles = program_features['title'][top_n]

    # For each of the top n workout programs, get out only specific columns and add each DF to a list
    progs = [huge_data[huge_data['title'] == i][metadata+info] for i in top_titles]
    return progs

def program_recommender(program, features, similarity_matrix, model, n_programs=5, within_cluster=False):
    """
    Takes in a user's inputted program vector or existing program index 
    and computes the top n similar workout programs.

    Args:
        program (int or list): If int, the index of an existing program to use as the query.
                               If list, a vector of numeric features followed by a string description
                               representing a custom user program.
        features (np.ndarray): Feature matrix of all programs (used for custom queries).
        model (SentenceTransformer): Model used to encode text descriptions (default: global model).
        n_programs (int): Number of similar programs to return (default: 5).

    Returns:
        list[pd.DataFrame]: List of DataFrames, each containing metadata and info for a recommended program.
    """
    metadata = ['title', 'description', 'level', 'goal', 'equipment', 'program_length','time_per_workout', 'number_of_exercises']
    info = ['week', 'day', 'exercise_name', 'sets', 'reps', 'intensity']

    if (type(program) == int):
        return find_top_n(similarity_matrix, 
                          n_programs, 
                          program, 
                          metadata, 
                          info, 
                          features['cluster'].iloc[program] if within_cluster else None,
                          features if within_cluster else None
        )
    elif (type(program) == list):
        # Encodes the user's description for the workout
        query_embd = model.encode(program[-1])
        query_numeric = np.array(program[:-1], dtype=np.float32)
        # Concatenate the numeric features and the embedding
        query_full = np.concatenate([query_numeric, query_embd.flatten()])

        # Standardize the query
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.drop(columns=['cluster']))
        query_full_scaled = (scaler.transform([query_full])[0].reshape(1, -1)) # Reshaping turns the query into a 2D array
        cluster = int(kmeans.predict(query_full_scaled))

        # Compute cosine similarity between the query and all existing (already scaled) features
        similarities_to_query = cosine_similarity(
            features_scaled,
            query_full_scaled 
        ).flatten()

        features_scaled = pd.concat([pd.DataFrame(features_scaled), features['cluster']], axis=1)

        return find_top_n(
            similarities_to_query.reshape(1, -1),
            n_programs,
            0,
            metadata,
            info,
            cluster if within_cluster else None,
            features_scaled if within_cluster else None
        )
    
    else:
        raise ValueError('Value inputted is not an int or NumPy array.')

@st.cache_resource
def dataset_setup(final_features, n_clusters=25, random_state=4):
    """
    Scales the final features and clusters them into n_clusters.
    """
    scaler = StandardScaler()
    final_features_scaled = pd.DataFrame(
        scaler.fit_transform(final_features),
        columns=final_features.columns,
        index=final_features.index
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clustering_data = final_features_scaled.copy()
    clustering_data['cluster'] = kmeans.fit_predict(clustering_data)

    return clustering_data, kmeans

clustering_data, kmeans = dataset_setup(final_features)
final_features.loc[:, 'cluster'] = clustering_data['cluster']

similarities = cosine_similarity(clustering_data)

# Header
st.markdown("""
<div class="main-header">
    <h1>ML Powered Workout Program Recommender</h1>
    <p>Discover your perfect workout program with machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional info
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    1. **Describe your goals** - Tell us what you're looking for
    2. **Set your preferences** - Choose intensity, equipment, etc.
    3. **Get recommendations** - We find the best programs for you
    """)
    
    st.markdown("### Dataset Info")
    st.metric("Total Programs", f"{len(huge_data['title'].unique()):,}")
    st.metric("Total Exercises", f"{len(huge_data):,}")
    
    st.markdown("### Features")
    st.markdown("""
    - **ML-Powered Matching**: Uses advanced fine-tuned ALBERT model to understand your goals
    - **Smart Clustering**: Groups similar programs for better recommendations using K-Means clustering
    - **Customizable Filters**: Specify your preferences
    - **Detailed Programs**: Get complete workout plans with exercises
    """)

# Main form
st.markdown("### Create Your Perfect Workout Program")

with st.form("program_input_form"):
    # Program Description Section
    st.markdown("#### Describe Your Goals")
    description_query = st.text_area(
        'What kind of workout program are you looking for?',
        placeholder='Examples: "Insane Arnold program, not for the weak. Max intensity in every exercise. Push to failure ALWAYS."\n"I want a challenging bodybuilding program focused on muscle growth with compound movements. I prefer high intensity training with progressive overload."\n"Program focused on stability, mobility, and flexibility. Lower intensity, but still challenging."',
        height=100,
        help="Be as detailed as possible about your goals, preferences, and any specific requirements"
    )
    
    # Training Parameters Section
    st.markdown("#### Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        reps_count = st.number_input(
            "Mean reps per exercise",
            min_value=0, max_value=100, value=8, step=1,
            help="Average number of reps per rep-based exercise"
        )
        intensity = st.slider(
            "Intensity Level (1-10)",
            min_value=1, max_value=10, value=7, step=1,
            help="Average intensity level across the program"
        )
    with col2:
        sets = st.number_input(
            "Total sets per week",
            min_value=0, max_value=100, value=20, step=1,
            help="Total number of sets per week"
        )
        reps_per_week = st.number_input(
            "Total reps per week",
            min_value=0, max_value=10000, value=200, step=10,
            help="Total number of reps per week"
        )
    with col3:
        reps_time = st.number_input(
            "Time per exercise (min)",
            min_value=0.0, max_value=120.0, value=10.0, step=0.5,
            help="Average time per time-based exercise"
        )
        is_rep_based = st.slider(
            "Rep-based exercises (%)",
            min_value=0.0, max_value=1.0, value=0.8, step=0.05,
            help="Fraction of exercises that are rep-based"
        )
    
    # Program Structure Section
    st.markdown("#### Program Structure")
    col1, col2, col3 = st.columns(3)
    with col1:
        program_length = st.number_input(
            "Program length (weeks)",
            min_value=1, max_value=52, value=8, step=1,
            help="Length of the program in weeks"
        )
        time_per_workout = st.number_input(
            "Workout length (minutes)",
            min_value=10, max_value=300, value=60, step=5,
            help="Average workout duration in minutes"
        )
    with col2:
        level = st.multiselect(
            "Experience Level",
            options=["beginner", "novice", "intermediate", "advanced"],
            default=["intermediate"],
            help="Select one or more experience levels for the program"
        )
    with col3:
        goal = st.multiselect(
            "Training Goal",
            options=[
                "olympic_weightlifting",
                "muscle_&_sculpting",
                "bodyweight_fitness",
                "powerbuilding",
                "bodybuilding",
                "powerlifting",
                "athletics"
            ],
            default=["bodybuilding"],
            help="Select one or more goals for the program"
        )
    
    # Equipment Section
    st.markdown("#### Equipment Available")
    equipment = st.multiselect(
        "What equipment do you have access to?",
        options=[
            "at home",
            "dumbbell only",
            "full gym",
            "garage gym"
        ],
        default=["full gym"],
        help="Select one or more equipment types for the program"
    )
    
    # Recommendation Settings
    st.markdown("#### Recommendation Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_programs = st.number_input(
            "Number of recommendations",
            min_value=1, max_value=20, value=5, step=1,
            help="Number of similar programs to display"
        )
    with col2:
        st.write("")  # Add some spacing
        within_cluster = st.checkbox(
            "Within cluster only",
            value=False,
            help="Only recommend programs from the same cluster for more focused results"
        )
    with col3:
        st.write("")  # Add some spacing
        submitted = st.form_submit_button("Get Recommendations", use_container_width=True)

if submitted:
    with st.spinner("Finding your perfect workout programs..."):
        # One-hot encode 'level', 'goal', and 'equipment'
        level_options = ["beginner", "novice", "intermediate", "advanced"]
        goal_options = [
            "olympic_weightlifting",
            "muscle_&_sculpting",
            "bodyweight_fitness",
            "powerbuilding",
            "bodybuilding",
            "powerlifting",
            "athletics"
        ]
        equipment_options = [
            "at home",
            "dumbbell only",
            "full gym",
            "garage gym"
        ]

        level_onehot = [1 if l in level else 0 for l in level_options]
        goal_onehot = [1 if g in goal else 0 for g in goal_options]
        equipment_onehot = [1 if e in equipment else 0 for e in equipment_options]

        query = [
            reps_count, reps_time, is_rep_based, sets, reps_per_week, program_length, time_per_workout, intensity
        ] + level_onehot + goal_onehot + equipment_onehot + [description_query]

        programs = program_recommender(query, final_features, similarities, embedder, n_programs=n_programs, within_cluster=within_cluster)
    
    # Results Section
    st.markdown("## Your Recommended Workout Programs")
    
    for i, program in enumerate(programs):
        meta_cols = ['title', 'description', 'level', 'goal', 'equipment', 'program_length', 'time_per_workout', 'number_of_exercises']
        meta = program.iloc[0][meta_cols]

        def format_field(val):
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        return ", ".join(str(v).title() for v in parsed)
                except Exception:
                    pass
            
            # Handle formatting of possessive nouns and contractions
            if isinstance(val, str):
                # Split by spaces and handle each word
                words = val.split()
                formatted_words = []
                for word in words:
                    if "'" in word:
                        parts = word.split("'")
                        if len(parts) == 2:
                            formatted_word = parts[0].title() + "'" + parts[1].lower()
                        else:
                            formatted_word = word.title()
                    else:
                        formatted_word = word.title()
                    formatted_words.append(formatted_word)
                return " ".join(formatted_words)
            
            return str(val).title()

        # Create program card with title and metadata always visible
        st.markdown(f"""
        <div class="program-card">
            <div class="program-title">{format_field(meta['title'])}</div>
            <div class="program-meta">
                <div class="meta-item">
                    <span class="meta-label">Description:</span>
                    <span class="meta-value">{meta['description']}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Level:</span>
                    <span class="meta-value">{format_field(meta['level'])}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Goal:</span>
                    <span class="meta-value">{format_field(meta['goal'])}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Equipment:</span>
                    <span class="meta-value">{format_field(meta['equipment'])}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Duration:</span>
                    <span class="meta-value">{format_field(meta['program_length'])} weeks</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Workout Time:</span>
                    <span class="meta-value">{format_field(meta['time_per_workout'])} min</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Exercises:</span>
                    <span class="meta-value">{format_field(meta['number_of_exercises'])} total</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Exercise details in expander
        with st.expander(f"View Complete Workout Plan - {format_field(meta['title'])}", expanded=False):
            exercise_cols = ['week', 'day', 'exercise_name', 'reps', 'sets', 'intensity']
            program_exercises = program[exercise_cols]
            
            # Add some styling to the dataframe
            st.markdown("#### Exercise Breakdown")
            st.dataframe(
                program_exercises,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="Download Program",
                    data=program_exercises.to_csv(index=False).encode('utf-8'),
                    file_name=f"{format_field(meta['title']).replace(' ', '_')}_program.csv",
                    mime='text/csv',
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
    <p>Built by Atherv Vidhate</p>
</div>
""", unsafe_allow_html=True)