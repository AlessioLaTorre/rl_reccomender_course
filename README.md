## Overview

This project implements a reinforcement learning framework for **sequential, job-oriented course recommendation**, with explicit modeling of skill mastery and personalized career objectives.

The core research question addressed is:

> How can we recommend an ordered sequence of courses that reduces skill gaps toward relevant jobs, while accounting for graded expertise levels and learner-specific goals?

Unlike traditional employability-based recommenders that optimize only the number of accessible jobs, this system:

- models skills with discrete mastery levels,
- explicitly measures skill-gap reduction,
- formulates reward signals grounded in the **Usefulness of Information** principle,
- supports preference-conditioned goal filtering (wanted / avoided skills),
- enforces prerequisite constraints via action masking.

The reinforcement learning environment, reward formulations, and goal-conditioning mechanisms have been redesigned and restructured compared to prior job-oriented course recommendation frameworks.

This repository contains the full implementation used for experimentation and thesis research.

## Project Structure

The repository is organized into three main components:
- a **Chatbot module** with GUI and conversational logic
- a **Reinforcement Learning backend** (UIR)
- supporting datasets and experimental components


```text
rl_recommender_course/
├── Chatbot/                     # Conversational interface and GUI
│   ├── NEW_GUI.py               # Streamlit GUI (main demo entry point)
│   ├── chat_handler.py          # Conversation handling and state updates
│   ├── LLMDialogManager.py      # LLM wrapper (Mistral v2)
│   ├── chatbot.py               # Terminal-based chatbot (legacy)
│   ├── state.py                 # User preference state
│   ├── learnerProfile.py        # Learner skill profile representation
│   ├── taxonomy_index.py        # ESCO taxonomy indexing
│   ├── data_loader.py           # Dataset loading utilities
│   ├── utils.py                 # Helper functions
│   │
│   ├── CV_pdf/                  # Example resumes (PDF)
│   │
│   ├── Embeddings/              # Skill embedding and semantic search
│   │   ├── build_skill_embeddings.py
│   │   ├── skill_search.py
│   │   ├── E_skills.npy
│   │   └── uids.npy
│   │
│   └── NER/                     # Named Entity Recognition (skills)
│       ├── BIO.ipynb            # NER experimentation notebook
│       ├── training_data_*.json
│       ├── dataset*.json
│       └── pretrained checkpoints
│
├── UIR/                         # Reinforcement Learning backend (main)
│   ├── Scripts/
│   │   ├── CourseRecEnv.py      # RL environment
│   │   ├── Dataset.py           # Dataset handling
│   │   ├── Reinforce.py         # RL agent logic
│   │   ├── pipeline.py          # Training pipeline entry point
│   │   ├── matchings.py         # Skill-job matching utilities
│   │   ├── tuning.py            # Experimental reward-weight tuning (beta1)
│   │   └── evaluation.py        # results visualization and evaluation
│   │    
│   │
│   ├── Optimization/            # PPO and MaskablePPO hyperparameter tuning
│   │   ├── tuning_ready.py      # Optuna tuning entry point
│   │   └── README.md            # Tuning instructions and options
│   │
│   ├── config/
│   │   └── run.yaml             # Training and inference configuration
│   │
│   ├── models_weights/          # Pretrained RL models (not versioned)
│   └── results/                 # Training outputs and plots
│
├── Data-Collection/
│   └── Final/                   # Datasets
│       ├── courses.json
│       ├── jobs.json
│       ├── resumes.json
│       ├── taxonomy.csv
│       └── mastery_levels.json
│
├── requirements.txt             # Project dependencies
├── README.md
└── LICENSE
```

## Installation (Windows)

We recommend using a Python virtual environment.
Supported Python version: 3.12

### 1. Create and activate a virtual environment
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch

CUDA (recommended for LLM inference):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
CPU-only (Not Recommended for LLM Inference)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Project Dependencies
```
pip install -r requirements.txt
```


## Running the Demo (Streamlit GUI)

The Streamlit GUI is the **main entry point** for the project demo.

### Important prerequisite

Before running the demo, you must first train the required models:

- **Reinforcement Learning model(s):**  
  you need to train an RL model **for each value of `k`** you want to use for recommendation. 
  Where k denotes the number of courses the agent will recommend as a sequence 
  The training procedure is described later in the section  
  **Training a Reinforcement Learning Model**.

- **Named Entity Recognition (NER) model:**  
  you must train the NER model by running the notebook  
  `Chatbot/NER/BIO.ipynb`  
  from start to finish as it is.

The demo depends on these trained models and will not work correctly unless they are available.

### Start the GUI

From the project root directory (`rl_recommender_course`):

```bash
streamlit run Chatbot/NEW_GUI.py
```

Demo Capabilities

Through the GUI, users can:

- interact with the chatbot in natural language

- specify skills they want to acquire or avoid

- load a resume (CV) in PDF format

- receive personalized course recommendations

- obtain explanations for both preferences and recommendations



---


## Chatbot Commands

The chatbot supports the following commands:

- `:sem <text>`  
  Extracts skill preferences from natural language input and updates the user profile.

- `:rec`  
  Generates a personalized sequence of recommended courses.

- `:myskills`  
  Displays the skills currently associated with the user profile.

- `:show`  
  Shows current include and avoid skill preferences.

- `load resume`  
  Extracts skills from a resume (PDF) and updates the user profile.

- `clear`  
  Resets the user profile and clears all preferences.

which are automatically computed when using streamlit with specific buttons


## Reinforcement Learning Backend (UIR)

The **UIR (Usefulness-based Information Reward)** approach is the primary reinforcement learning
method used in this project.

### Key Characteristics

- **Algorithm**: PPO / MaskablePPO  
- **State**: learner skill vector and preferences  
- **Action space**: available courses  
- **Reward**: usefulness-based (skill acquisition and job Employability)

Pretrained models are not included in the repository since they are too large.



## Training a Reinforcement Learning Model

Training a new reinforcement learning model is **required** to run the demo.

### Configure Training

Edit the configuration file:

`UIR/config/run.yaml`


This file controls:

- **Data paths**: taxonomy, courses, jobs, learner profiles, mastery levels, and result directories.
- **Dataset construction**: number of courses, jobs, learner profiles, taxonomy level (L3 vs L4), and dataset seed for reproducibility.
- **Model selection**: algorithm (DQN, PPO, A2C, MaskablePPO), reward feature (Employability / UIR / EUIR), and gap/threshold method.
- **Training configuration**: total training steps, evaluation frequency, random seed.
- **Hyperparameters**: optimizer settings, learning rate schedule, entropy coefficient, clipping range, and KL target.
- **Preference conditioning**: whether to enable goal-based filtering in the environment.
- **Pretrained model usage**: load/save behavior and experiment naming.

### Run Training
```
python -m UIR.Scripts.pipeline --config UIR/config/run.yaml
```

Training outputs, logs, and models are saved according to the configuration.

#### Hyperparameter Optimization

Optuna-based hyperparameter tuning for PPO and MaskablePPO is available in
`UIR/Optimization/tuning_ready.py`, for the selected reward formulations and values of `k`.

This is separate from the experimental `UIR/Scripts/tuning.py`, currently configured
to optimize the reward weight `beta1`.

For configuration and run instructions, see the [Optimization README](UIR/Optimization/README.md).

Hyperparameter tuning is optional and is not required for standard usage.


---



## Models

### Named Entity Recognition (NER).
- The model is used to extract skill mentions from user input.
- The model path is configured directly in `chat_handler.py`.

You should train the NER model by running BIO.ipynb notebook before using it

### Large Language Model (LLM)

- The chatbot uses an LLM downloaded automatically from HuggingFace.
- Default model:
  - `mistralai/Mistral-7B-Instruct-v0.2`
- The first run may take several minutes due to model download.

### Reinforcement Learning Models

- Pretrained RL models are stored in:
  `UIR/models_weights/`

You should train an RL agent before using it.
To decide which model to use, you must change run.yaml.


---

## Hardware Notes

- A GPU is **strongly recommended** for running the chatbot with the LLM.
- CPU-only execution is supported but will be significantly slower.
- Reinforcement learning training primarily runs on CPU.


## Demo
🎥 Video walkthrough: <https://drive.google.com/file/d/1pfKi74UfCfmA7jmxe55IWCslMnUdnalt/view?usp=sharing>


## Notes

- This project is intended for **research and demonstration purposes**.
- Code clarity and modularity are prioritized over production-level optimization.
- Detailed methodological explanations are provided in the associated publication.




## Acknowledgments

This work builds upon the job-oriented course recommendation framework introduced by Jibril Frej in JCRec:

- Frej, J., Dai, A., Montariol, S., Bosselut, A., & Käser, T. (2024).  
  *Course Recommender Systems Need to Consider the Job Market*.  
  Proceedings of SIGIR '24.  
  https://doi.org/10.1145/3626772.3657847
- GitHub: https://github.com/Jibril-Frej/JCRec

Early development was based on Mark’s extension of this framework:

- WUIR-CLASS-recSys: https://github.com/bm1nhtr/WUIR-CLASS-recSys

The current implementation substantially restructures the reinforcement learning environment, reward modeling, and training pipeline, and does not rely on the original CLASS-based approach.