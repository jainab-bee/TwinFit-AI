
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher

SKILL_TAXONOMY: Dict[str, Dict] = {
    "Python": {
        "category": "Programming Language",
        "aliases": ["python3", "python programming", "python language"],
        "related_skills": ["R", "Julia", "Scala"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Java": {
        "category": "Programming Language",
        "aliases": ["java programming", "core java"],
        "related_skills": ["Kotlin", "Scala", "C++"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "C++": {
        "category": "Programming Language",
        "aliases": ["cpp", "c plus plus", "c++ programming"],
        "related_skills": ["C", "Java", "Rust"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "C": {
        "category": "Programming Language",
        "aliases": ["c programming", "c language"],
        "related_skills": ["C++", "Embedded Systems"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "JavaScript": {
        "category": "Programming Language",
        "aliases": ["javascript", "java script", "js", "ecmascript", "es6", "es2015"],
        "related_skills": ["TypeScript", "Node.js", "React"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": ["TypeScript"]
    },
    "TypeScript": {
        "category": "Programming Language",
        "aliases": ["typescript", "ts"],
        "related_skills": ["JavaScript"],
        "interchangeable_skills": [],
        "broader_skill": "JavaScript",
        "narrower_skills": []
    },
    "SQL": {
        "category": "Query Language",
        "aliases": ["structured query language", "sql query", "sql queries"],
        "related_skills": ["PostgreSQL", "MySQL", "SQLite", "NoSQL"],
        "interchangeable_skills": [],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "R": {
        "category": "Programming Language",
        "aliases": ["r programming", "r language", "r statistical"],
        "related_skills": ["Python", "Statistics", "Data Analysis"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Go": {
        "category": "Programming Language",
        "aliases": ["golang", "go language", "go programming"],
        "related_skills": ["Rust", "C++", "Docker"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Rust": {
        "category": "Programming Language",
        "aliases": ["rust programming", "rust language"],
        "related_skills": ["C++", "Go", "Systems Programming"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Kotlin": {
        "category": "Programming Language",
        "aliases": ["kotlin programming"],
        "related_skills": ["Java", "Android Development"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Scala": {
        "category": "Programming Language",
        "aliases": ["scala programming"],
        "related_skills": ["Java", "Apache Spark", "Functional Programming"],
        "interchangeable_skills": [],
        "broader_skill": "Programming",
        "narrower_skills": []
    },
    "Shell Scripting": {
        "category": "Scripting",
        "aliases": ["bash", "bash scripting", "shell script", "bash script", "sh scripting"],
        "related_skills": ["Linux", "DevOps", "Automation"],
        "interchangeable_skills": [],
        "broader_skill": "Scripting",
        "narrower_skills": []
    },
    "Machine Learning": {
        "category": "AI/ML",
        "aliases": ["ml", "machine-learning", "applied machine learning"],
        "related_skills": [
            "Deep Learning", "Statistical Modeling", "Data Science",
            "Feature Engineering", "Model Evaluation"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Artificial Intelligence",
        "narrower_skills": [
            "Supervised Learning", "Unsupervised Learning",
            "Deep Learning", "Feature Engineering"
        ]
    },
    "Deep Learning": {
        "category": "AI/ML",
        "aliases": ["dl", "deep neural learning"],
        "related_skills": [
            "Machine Learning", "Neural Networks",
            "TensorFlow", "PyTorch", "Keras"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": [
            "Convolutional Neural Networks", "Recurrent Neural Networks",
            "Transfer Learning", "Transformers"
        ]
    },
    "Artificial Intelligence": {
        "category": "AI/ML",
        "aliases": ["ai", "artificial-intelligence"],
        "related_skills": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision"],
        "interchangeable_skills": [],
        "broader_skill": "",
        "narrower_skills": ["Machine Learning", "Deep Learning", "Natural Language Processing"]
    },
    "Natural Language Processing": {
        "category": "AI/ML",
        "aliases": [
            "nlp", "natural-language processing", "text processing",
            "natural language understanding", "nlu"
        ],
        "related_skills": [
            "Deep Learning", "Hugging Face Transformers", "spaCy",
            "NLTK", "Text Classification", "Sentiment Analysis"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": ["Text Classification", "Sentiment Analysis", "Named Entity Recognition"]
    },
    "Computer Vision": {
        "category": "AI/ML",
        "aliases": [
            "cv", "image processing", "visual recognition",
            "image recognition", "object detection"
        ],
        "related_skills": [
            "Deep Learning", "OpenCV", "Convolutional Neural Networks",
            "TensorFlow", "PyTorch"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": ["Object Detection", "Image Segmentation"]
    },
    "Neural Networks": {
        "category": "AI/ML",
        "aliases": [
            "neural network", "artificial neural network",
            "artificial neural networks", "ann"
        ],
        "related_skills": ["Deep Learning", "TensorFlow", "PyTorch", "Keras"],
        "interchangeable_skills": [],
        "broader_skill": "Deep Learning",
        "narrower_skills": [
            "Convolutional Neural Networks", "Recurrent Neural Networks"
        ]
    },
    "Convolutional Neural Networks": {
        "category": "AI/ML",
        "aliases": ["convolutional neural network", "cnn", "cnns"],
        "related_skills": ["Computer Vision", "TensorFlow", "PyTorch", "Image Classification"],
        "interchangeable_skills": [],
        "broader_skill": "Neural Networks",
        "narrower_skills": []
    },
    "Recurrent Neural Networks": {
        "category": "AI/ML",
        "aliases": ["recurrent neural network", "rnn", "rnns", "lstm", "gru"],
        "related_skills": [
            "Natural Language Processing", "Time Series Analysis",
            "TensorFlow", "PyTorch"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Neural Networks",
        "narrower_skills": []
    },
    "Transfer Learning": {
        "category": "AI/ML",
        "aliases": ["transfer-learning", "fine-tuning", "fine tuning"],
        "related_skills": ["Deep Learning", "Hugging Face Transformers", "PyTorch", "TensorFlow"],
        "interchangeable_skills": [],
        "broader_skill": "Deep Learning",
        "narrower_skills": []
    },
    "Feature Engineering": {
        "category": "AI/ML",
        "aliases": ["feature-engineering", "feature selection", "feature extraction"],
        "related_skills": [
            "Machine Learning", "Data Preprocessing", "scikit-learn", "Pandas"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Model Deployment": {
        "category": "MLOps",
        "aliases": ["ml model deployment", "deploying machine learning models", "model serving"],
        "related_skills": ["Docker", "Kubernetes", "FastAPI", "Flask", "MLflow"],
        "interchangeable_skills": [],
        "broader_skill": "MLOps",
        "narrower_skills": []
    },
    "Model Evaluation": {
        "category": "AI/ML",
        "aliases": [
            "machine learning model evaluation", "model metrics",
            "evaluation metrics", "model performance"
        ],
        "related_skills": ["Machine Learning", "scikit-learn", "Cross Validation"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Supervised Learning": {
        "category": "AI/ML",
        "aliases": ["supervised machine learning", "classification", "regression"],
        "related_skills": ["Machine Learning", "scikit-learn", "Feature Engineering"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Unsupervised Learning": {
        "category": "AI/ML",
        "aliases": ["unsupervised machine learning", "clustering", "dimensionality reduction"],
        "related_skills": ["Machine Learning", "scikit-learn", "Data Analysis"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Reinforcement Learning": {
        "category": "AI/ML",
        "aliases": ["rl", "deep reinforcement learning", "drl"],
        "related_skills": ["Machine Learning", "Deep Learning", "PyTorch"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Statistical Modeling": {
        "category": "AI/ML",
        "aliases": ["statistical analysis", "statistics", "probabilistic modeling"],
        "related_skills": ["Machine Learning", "R", "Python", "NumPy", "SciPy"],
        "interchangeable_skills": [],
        "broader_skill": "Data Science",
        "narrower_skills": []
    },
    "Time Series Analysis": {
        "category": "AI/ML",
        "aliases": [
            "time series forecasting", "time-series", "forecasting",
            "arima", "prophet"
        ],
        "related_skills": ["Machine Learning", "Statistical Modeling", "Pandas"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Generative AI": {
        "category": "AI/ML",
        "aliases": [
            "genai", "gen ai", "generative models", "llm",
            "large language models", "large language model"
        ],
        "related_skills": [
            "Natural Language Processing", "Hugging Face Transformers",
            "Deep Learning", "Prompt Engineering"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Artificial Intelligence",
        "narrower_skills": []
    },
    "Prompt Engineering": {
        "category": "AI/ML",
        "aliases": ["prompt design", "llm prompting", "chain of thought"],
        "related_skills": ["Generative AI", "Natural Language Processing"],
        "interchangeable_skills": [],
        "broader_skill": "Generative AI",
        "narrower_skills": []
    },

    "scikit-learn": {
        "category": "ML Framework",
        "aliases": [
            "scikit learn", "scikit", "sklearn", "sci-kit learn",
            "scikit-learn library"
        ],
        "related_skills": [
            "Machine Learning", "Feature Engineering", "Model Evaluation",
            "Python"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "TensorFlow": {
        "category": "Deep Learning Framework",
        "aliases": ["tensorflow", "tensor flow", "tf", "tensorflow 2"],
        "related_skills": ["PyTorch", "Keras", "Deep Learning", "Neural Networks"],
        "interchangeable_skills": [],
        "broader_skill": "Deep Learning",
        "narrower_skills": ["Keras"]
    },
    "PyTorch": {
        "category": "Deep Learning Framework",
        "aliases": ["pytorch", "py torch", "torch"],
        "related_skills": ["TensorFlow", "Keras", "Deep Learning", "Neural Networks"],
        "interchangeable_skills": [],
        "broader_skill": "Deep Learning",
        "narrower_skills": []
    },
    "Keras": {
        "category": "Deep Learning Framework",
        "aliases": ["keras api", "keras library"],
        "related_skills": ["TensorFlow", "PyTorch", "Deep Learning"],
        "interchangeable_skills": [],
        "broader_skill": "TensorFlow",
        "narrower_skills": []
    },
    "XGBoost": {
        "category": "ML Framework",
        "aliases": ["xgboost", "xg boost", "extreme gradient boosting"],
        "related_skills": ["LightGBM", "scikit-learn", "Gradient Boosting", "Machine Learning"],
        "interchangeable_skills": ["LightGBM"],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "LightGBM": {
        "category": "ML Framework",
        "aliases": ["lightgbm", "light gbm", "lgbm"],
        "related_skills": ["XGBoost", "scikit-learn", "Gradient Boosting", "Machine Learning"],
        "interchangeable_skills": ["XGBoost"],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },
    "Hugging Face Transformers": {
        "category": "NLP Framework",
        "aliases": [
            "hugging face", "huggingface", "transformers library",
            "huggingface transformers"
        ],
        "related_skills": [
            "Natural Language Processing", "Transfer Learning",
            "BERT", "GPT", "Deep Learning"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Natural Language Processing",
        "narrower_skills": []
    },
    "OpenCV": {
        "category": "Computer Vision Library",
        "aliases": ["opencv", "open cv", "cv2", "opencv2", "opencv4"],
        "related_skills": ["Computer Vision", "Image Processing", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Computer Vision",
        "narrower_skills": []
    },
    "spaCy": {
        "category": "NLP Library",
        "aliases": ["spacy", "spacy nlp"],
        "related_skills": ["Natural Language Processing", "NLTK", "Python"],
        "interchangeable_skills": ["NLTK"],
        "broader_skill": "Natural Language Processing",
        "narrower_skills": []
    },
    "NLTK": {
        "category": "NLP Library",
        "aliases": ["nltk", "natural language toolkit"],
        "related_skills": ["Natural Language Processing", "spaCy", "Python"],
        "interchangeable_skills": ["spaCy"],
        "broader_skill": "Natural Language Processing",
        "narrower_skills": []
    },
    "SciPy": {
        "category": "Scientific Computing",
        "aliases": ["scipy"],
        "related_skills": ["NumPy", "Statistical Modeling", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Data Science",
        "narrower_skills": []
    },
    "NumPy": {
        "category": "Data Science Library",
        "aliases": ["numpy", "np"],
        "related_skills": ["Pandas", "SciPy", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Data Science",
        "narrower_skills": []
    },
    "Pandas": {
        "category": "Data Science Library",
        "aliases": ["pandas", "pandas dataframe"],
        "related_skills": ["NumPy", "Data Preprocessing", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Data Science",
        "narrower_skills": []
    },
    "Matplotlib": {
        "category": "Data Visualization",
        "aliases": ["matplotlib", "pyplot"],
        "related_skills": ["Seaborn", "Plotly", "Data Analysis"],
        "interchangeable_skills": ["Seaborn"],
        "broader_skill": "Data Visualization",
        "narrower_skills": []
    },
    "Seaborn": {
        "category": "Data Visualization",
        "aliases": ["seaborn"],
        "related_skills": ["Matplotlib", "Plotly", "Data Analysis"],
        "interchangeable_skills": ["Matplotlib"],
        "broader_skill": "Data Visualization",
        "narrower_skills": []
    },
    "Plotly": {
        "category": "Data Visualization",
        "aliases": ["plotly", "plotly dash", "dash"],
        "related_skills": ["Matplotlib", "Seaborn", "Data Visualization"],
        "interchangeable_skills": [],
        "broader_skill": "Data Visualization",
        "narrower_skills": []
    },
    "Data Analysis": {
        "category": "Data Science",
        "aliases": [
            "data analytics", "data analysing", "data analysis",
            "data analysis and visualization"
        ],
        "related_skills": ["Pandas", "NumPy", "Exploratory Data Analysis"],
        "interchangeable_skills": [],
        "broader_skill": "Data Science",
        "narrower_skills": ["Exploratory Data Analysis"]
    },
    "Exploratory Data Analysis": {
        "category": "Data Science",
        "aliases": [
            "eda", "exploratory analysis",
            "exploratory data analysis"
        ],
        "related_skills": ["Data Analysis", "Pandas", "Matplotlib", "Seaborn"],
        "interchangeable_skills": [],
        "broader_skill": "Data Analysis",
        "narrower_skills": []
    },
    "Data Preprocessing": {
        "category": "Data Science",
        "aliases": [
            "data cleaning", "data preparation",
            "data preprocessing", "data wrangling"
        ],
        "related_skills": ["Feature Engineering", "Pandas", "scikit-learn"],
        "interchangeable_skills": [],
        "broader_skill": "Machine Learning",
        "narrower_skills": []
    },

    "FastAPI": {
        "category": "Backend Framework",
        "aliases": ["fastapi", "fast api"],
        "related_skills": ["Flask", "Django", "REST API", "Python", "Pydantic"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": []
    },
    "Flask": {
        "category": "Backend Framework",
        "aliases": ["flask", "flask framework"],
        "related_skills": ["FastAPI", "Django", "REST API", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": []
    },
    "Django": {
        "category": "Backend Framework",
        "aliases": ["django", "django rest framework", "drf"],
        "related_skills": ["Flask", "FastAPI", "REST API", "Python"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": []
    },
    "Node.js": {
        "category": "Backend Runtime",
        "aliases": ["node.js", "nodejs", "node js"],
        "related_skills": ["Express.js", "JavaScript", "REST API"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": ["Express.js"]
    },
    "Express.js": {
        "category": "Backend Framework",
        "aliases": ["express", "express.js", "expressjs"],
        "related_skills": ["Node.js", "JavaScript", "REST API"],
        "interchangeable_skills": [],
        "broader_skill": "Node.js",
        "narrower_skills": []
    },
    "REST API": {
        "category": "Backend",
        "aliases": [
            "rest api", "restful api", "restful apis",
            "restful services", "rest services", "api development"
        ],
        "related_skills": ["FastAPI", "Flask", "Django", "GraphQL"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": []
    },
    "GraphQL": {
        "category": "Backend",
        "aliases": ["graphql", "graph ql", "graph query language"],
        "related_skills": ["REST API", "API Development"],
        "interchangeable_skills": [],
        "broader_skill": "Backend Development",
        "narrower_skills": []
    },
    "Pydantic": {
        "category": "Python Library",
        "aliases": ["pydantic"],
        "related_skills": ["FastAPI", "Python", "Data Validation"],
        "interchangeable_skills": [],
        "broader_skill": "Python",
        "narrower_skills": []
    },
    "HTML": {
        "category": "Frontend",
        "aliases": ["html5", "hypertext markup language", "html/css"],
        "related_skills": ["CSS", "JavaScript", "React"],
        "interchangeable_skills": [],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },
    "CSS": {
        "category": "Frontend",
        "aliases": ["css3", "cascading style sheets"],
        "related_skills": ["HTML", "JavaScript", "Tailwind CSS", "Bootstrap"],
        "interchangeable_skills": [],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },
    "React": {
        "category": "Frontend Framework",
        "aliases": ["react.js", "reactjs", "react js"],
        "related_skills": ["JavaScript", "TypeScript", "Redux", "Node.js"],
        "interchangeable_skills": [],
        "broader_skill": "Frontend Development",
        "narrower_skills": ["Redux"]
    },
    "Redux": {
        "category": "Frontend Library",
        "aliases": ["redux", "redux toolkit", "react redux"],
        "related_skills": ["React", "JavaScript"],
        "interchangeable_skills": [],
        "broader_skill": "React",
        "narrower_skills": []
    },
    "Bootstrap": {
        "category": "Frontend Framework",
        "aliases": ["bootstrap", "bootstrap css"],
        "related_skills": ["CSS", "HTML", "Tailwind CSS"],
        "interchangeable_skills": ["Tailwind CSS"],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },
    "Tailwind CSS": {
        "category": "Frontend Framework",
        "aliases": ["tailwind", "tailwindcss", "tailwind css"],
        "related_skills": ["CSS", "HTML", "Bootstrap"],
        "interchangeable_skills": ["Bootstrap"],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },
    "Vue.js": {
        "category": "Frontend Framework",
        "aliases": ["vue", "vuejs", "vue.js"],
        "related_skills": ["React", "Angular", "JavaScript"],
        "interchangeable_skills": [],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },
    "Angular": {
        "category": "Frontend Framework",
        "aliases": ["angular", "angularjs", "angular.js"],
        "related_skills": ["React", "Vue.js", "TypeScript", "JavaScript"],
        "interchangeable_skills": [],
        "broader_skill": "Frontend Development",
        "narrower_skills": []
    },

    "MySQL": {
        "category": "Database",
        "aliases": ["mysql", "my sql"],
        "related_skills": ["PostgreSQL", "SQLite", "SQL"],
        "interchangeable_skills": ["PostgreSQL", "SQLite"],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "PostgreSQL": {
        "category": "Database",
        "aliases": ["postgresql", "postgres", "postgre sql", "psql"],
        "related_skills": ["MySQL", "SQLite", "SQL"],
        "interchangeable_skills": ["MySQL"],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "MongoDB": {
        "category": "Database",
        "aliases": ["mongodb", "mongo db", "mongo"],
        "related_skills": ["Redis", "NoSQL", "Database Management Systems"],
        "interchangeable_skills": [],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "SQLite": {
        "category": "Database",
        "aliases": ["sqlite", "sqlite3"],
        "related_skills": ["MySQL", "PostgreSQL", "SQL"],
        "interchangeable_skills": ["MySQL"],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "Redis": {
        "category": "Database",
        "aliases": ["redis", "redis cache"],
        "related_skills": ["MongoDB", "Caching", "Database Management Systems"],
        "interchangeable_skills": [],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "Database Management Systems": {
        "category": "Database",
        "aliases": [
            "dbms", "database management system", "database systems",
            "relational databases"
        ],
        "related_skills": ["SQL", "MySQL", "PostgreSQL"],
        "interchangeable_skills": [],
        "broader_skill": "Database",
        "narrower_skills": ["SQL", "MySQL", "PostgreSQL", "MongoDB"]
    },
    "Elasticsearch": {
        "category": "Database",
        "aliases": ["elastic search", "elasticsearch"],
        "related_skills": ["Database Management Systems", "Search", "Kibana"],
        "interchangeable_skills": [],
        "broader_skill": "Database",
        "narrower_skills": []
    },
    "Docker": {
        "category": "DevOps",
        "aliases": [
            "docker", "docker container", "docker containers",
            "containerization", "dockerfile"
        ],
        "related_skills": ["Kubernetes", "CI/CD", "DevOps", "Linux"],
        "interchangeable_skills": [],
        "broader_skill": "DevOps",
        "narrower_skills": []
    },
    "Kubernetes": {
        "category": "DevOps",
        "aliases": ["kubernetes", "k8s"],
        "related_skills": ["Docker", "CI/CD", "DevOps", "Cloud"],
        "interchangeable_skills": [],
        "broader_skill": "DevOps",
        "narrower_skills": []
    },
    "Git": {
        "category": "Developer Tool",
        "aliases": ["git", "git version control"],
        "related_skills": ["GitHub", "GitLab", "Version Control"],
        "interchangeable_skills": [],
        "broader_skill": "Version Control",
        "narrower_skills": []
    },
    "GitHub": {
        "category": "Developer Tool",
        "aliases": ["github", "git hub"],
        "related_skills": ["Git", "GitLab", "CI/CD"],
        "interchangeable_skills": ["GitLab"],
        "broader_skill": "Version Control",
        "narrower_skills": []
    },
    "GitLab": {
        "category": "Developer Tool",
        "aliases": ["gitlab", "git lab"],
        "related_skills": ["Git", "GitHub", "CI/CD"],
        "interchangeable_skills": ["GitHub"],
        "broader_skill": "Version Control",
        "narrower_skills": []
    },
    "CI/CD": {
        "category": "DevOps",
        "aliases": [
            "ci/cd", "continuous integration", "continuous deployment",
            "continuous delivery", "cicd", "ci cd",
            "github actions", "jenkins"
        ],
        "related_skills": ["Docker", "Kubernetes", "DevOps", "Git"],
        "interchangeable_skills": [],
        "broader_skill": "DevOps",
        "narrower_skills": []
    },
    "Amazon Web Services": {
        "category": "Cloud",
        "aliases": [
            "aws", "amazon web services", "aws cloud",
            "aws s3", "aws ec2", "aws lambda"
        ],
        "related_skills": ["Microsoft Azure", "Google Cloud Platform", "Docker", "Kubernetes"],
        "interchangeable_skills": [],
        "broader_skill": "Cloud",
        "narrower_skills": []
    },
    "Microsoft Azure": {
        "category": "Cloud",
        "aliases": ["azure", "microsoft azure", "azure cloud"],
        "related_skills": ["Amazon Web Services", "Google Cloud Platform", "DevOps"],
        "interchangeable_skills": [],
        "broader_skill": "Cloud",
        "narrower_skills": []
    },
    "Google Cloud Platform": {
        "category": "Cloud",
        "aliases": [
            "gcp", "google cloud", "google cloud platform",
            "google cloud services"
        ],
        "related_skills": ["Amazon Web Services", "Microsoft Azure", "DevOps"],
        "interchangeable_skills": [],
        "broader_skill": "Cloud",
        "narrower_skills": []
    },
    "Linux": {
        "category": "Operating System",
        "aliases": ["linux", "unix/linux", "ubuntu", "unix", "centos", "debian"],
        "related_skills": ["Shell Scripting", "Docker", "DevOps"],
        "interchangeable_skills": [],
        "broader_skill": "Operating Systems",
        "narrower_skills": []
    },
    "Terraform": {
        "category": "DevOps",
        "aliases": ["terraform", "infrastructure as code", "iac"],
        "related_skills": ["Amazon Web Services", "Google Cloud Platform", "DevOps", "Kubernetes"],
        "interchangeable_skills": [],
        "broader_skill": "DevOps",
        "narrower_skills": []
    },

    "Data Structures and Algorithms": {
        "category": "Computer Science",
        "aliases": [
            "dsa", "data structures", "data structures & algorithms",
            "data structures and algorithms", "algorithms"
        ],
        "related_skills": ["Object-Oriented Programming", "Python", "Java"],
        "interchangeable_skills": [],
        "broader_skill": "Computer Science",
        "narrower_skills": []
    },
    "Object-Oriented Programming": {
        "category": "Computer Science",
        "aliases": [
            "oop", "oops", "object oriented programming",
            "object-oriented programming", "object-oriented design"
        ],
        "related_skills": [
            "Data Structures and Algorithms", "Design Patterns",
            "Java", "Python", "C++"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Computer Science",
        "narrower_skills": []
    },
    "Operating Systems": {
        "category": "Computer Science",
        "aliases": [
            "operating system", "operating systems",
            "os concepts", "os fundamentals"
        ],
        "related_skills": ["Linux", "Computer Networks", "Data Structures and Algorithms"],
        "interchangeable_skills": [],
        "broader_skill": "Computer Science",
        "narrower_skills": ["Linux"]
    },
    "Computer Networks": {
        "category": "Computer Science",
        "aliases": [
            "computer networking", "computer networks",
            "networking fundamentals", "network protocols",
            "tcp/ip"
        ],
        "related_skills": ["Operating Systems", "Docker", "Cloud"],
        "interchangeable_skills": [],
        "broader_skill": "Computer Science",
        "narrower_skills": []
    },
    "Software Development": {
        "category": "Software Engineering",
        "aliases": ["software engineering", "software development"],
        "related_skills": [
            "Object-Oriented Programming", "Agile",
            "Data Structures and Algorithms"
        ],
        "interchangeable_skills": [],
        "broader_skill": "Computer Science",
        "narrower_skills": []
    },
    "Agile": {
        "category": "Methodology",
        "aliases": ["agile methodology", "agile development", "agile practices"],
        "related_skills": ["Scrum", "Software Development", "Project Management"],
        "interchangeable_skills": ["Scrum"],
        "broader_skill": "Software Development",
        "narrower_skills": ["Scrum"]
    },
    "Scrum": {
        "category": "Methodology",
        "aliases": ["scrum methodology", "scrum framework"],
        "related_skills": ["Agile", "Software Development"],
        "interchangeable_skills": ["Agile"],
        "broader_skill": "Agile",
        "narrower_skills": []
    },
    "Design Patterns": {
        "category": "Software Engineering",
        "aliases": ["software design patterns", "design pattern"],
        "related_skills": ["Object-Oriented Programming", "Software Development"],
        "interchangeable_skills": [],
        "broader_skill": "Software Engineering",
        "narrower_skills": []
    },

    "Streamlit": {
        "category": "Application Framework",
        "aliases": ["streamlit"],
        "related_skills": ["Python", "Gradio", "Data Visualization"],
        "interchangeable_skills": ["Gradio"],
        "broader_skill": "Application Framework",
        "narrower_skills": []
    },
    "Gradio": {
        "category": "Application Framework",
        "aliases": ["gradio"],
        "related_skills": ["Python", "Streamlit", "Machine Learning"],
        "interchangeable_skills": ["Streamlit"],
        "broader_skill": "Application Framework",
        "narrower_skills": []
    },
    "MLflow": {
        "category": "MLOps",
        "aliases": ["mlflow", "ml flow"],
        "related_skills": ["Model Deployment", "Docker", "Experiment Tracking"],
        "interchangeable_skills": [],
        "broader_skill": "MLOps",
        "narrower_skills": []
    },
    "Apache Airflow": {
        "category": "Data Engineering",
        "aliases": ["airflow", "apache airflow"],
        "related_skills": ["Data Engineering", "ETL", "Python", "Docker"],
        "interchangeable_skills": [],
        "broader_skill": "Data Engineering",
        "narrower_skills": []
    },
    "Apache Spark": {
        "category": "Data Engineering",
        "aliases": [
            "spark", "apache spark", "pyspark", "py spark",
            "spark streaming"
        ],
        "related_skills": ["Hadoop", "Data Engineering", "Python", "Scala"],
        "interchangeable_skills": [],
        "broader_skill": "Data Engineering",
        "narrower_skills": []
    },
    "Hadoop": {
        "category": "Data Engineering",
        "aliases": ["hadoop", "apache hadoop", "hdfs", "mapreduce"],
        "related_skills": ["Apache Spark", "Data Engineering"],
        "interchangeable_skills": [],
        "broader_skill": "Data Engineering",
        "narrower_skills": []
    },
}


NOISE_TERMS: Set[str] = {
    "skill", "skills", "technical", "technical skills",
    "language", "languages", "tool", "tools",
    "technology", "technologies", "framework", "frameworks",
    "library", "libraries", "experience", "knowledge",
    "understanding", "familiarity", "proficiency", "ability",
    "candidate", "developer", "engineer", "engineering",
    "machine learning engineer", "software engineer",
    "data scientist", "ai engineer", "ml engineer",
    "role", "job", "position", "responsibility", "responsibilities",
    "requirement", "requirements", "qualification", "qualifications",
    "project", "projects", "work", "team", "company", "business",
    "product", "application", "applications", "system", "systems",
    "model", "models", "feature", "features", "training", "learning",
    "learn", "machine", "development", "problem", "solution", "solutions",
    "computer science engineering", "summary", "objective", "profile",
    "education", "certification", "certifications", "responsibilities",
    "information", "background", "management", "design",
}

ROLE_PATTERNS: List[str] = [
    r"\b(?:an?|the)\s+(?:senior\s+|junior\s+|lead\s+|principal\s+)?"
    r"(?:machine learning|software|data|ai|backend|frontend|full[- ]stack|"
    r"deep learning|nlp|ml|devops|cloud)\s+(?:engineer|developer|scientist|analyst|researcher)\b",

    r"\b(?:machine learning|software|data|ai|backend|frontend|full[- ]stack|"
    r"deep learning|nlp|ml|devops|cloud)\s+(?:engineer|developer|scientist|analyst|researcher)\b",
]

SECTION_EVIDENCE_WEIGHT: Dict[str, float] = {
    "Work Experience": 1.0,
    "Projects": 0.85,
    "Technical Skills": 0.70,
    "Certifications": 0.60,
    "Summary": 0.50,
    "Education": 0.40,
    "Unknown": 0.30,
}

REQUIREMENT_WEIGHTS: Dict[str, float] = {
    "mandatory": 1.0,
    "preferred": 0.6,
    "optional": 0.3,
}

MATCH_SCORES: Dict[str, float] = {
    "exact_match": 1.0,
    "alias_match": 1.0,
    "related_match": 0.35,
    "missing": 0.0,
}

FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "mandatory_skill_coverage": 0.50,
    "preferred_skill_coverage": 0.20,
    "evidence_strength": 0.15,
    "project_experience_relevance": 0.10,
    "education_or_certification_relevance": 0.05,
}
_MANDATORY_KEYWORDS: List[str] = [
    "required", "must have", "must-have", "should have", "should-have",
    "mandatory", "minimum requirement", "minimum requirements",
    "strong experience in", "strong experience with",
    "proficiency in", "proficiency with", "solid experience",
    "hands-on experience", "extensive experience",
    "working knowledge of", "expertise in",
]

_PREFERRED_KEYWORDS: List[str] = [
    "preferred", "nice to have", "nice-to-have", "good to have",
    "good-to-have", "bonus", "plus", "desirable", "advantage",
    "ideally", "beneficial",
]

_OPTIONAL_KEYWORDS: List[str] = [
    "exposure to", "familiarity with", "basic knowledge of",
    "basic understanding of", "awareness of", "some experience with",
    "introductory knowledge",
]

@dataclass
class ExtractedSkill:
    """A skill found in a resume or JD, with full evidence metadata."""
    skill: str
    category: str
    match_type: str          # taxonomy_exact | alias_exact | semantic
    source_text: str
    source_section: str = "Unknown"
    confidence: float = 1.0
    evidence_weight: float = SECTION_EVIDENCE_WEIGHT["Unknown"]
    occurrence_count: int = 1
    sections_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "skill": self.skill,
            "category": self.category,
            "match_type": self.match_type,
            "source_text": self.source_text,
            "source_section": self.source_section,
            "confidence": self.confidence,
            "evidence_weight": self.evidence_weight,
            "occurrence_count": self.occurrence_count,
            "sections_found": self.sections_found,
        }


@dataclass
class JDSkill:
    """A skill requirement from the JD, classified by requirement type."""
    skill: str
    category: str
    requirement_type: str    # mandatory | preferred | optional
    importance_weight: float
    source_text: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SkillMatchResult:
    """Outcome of matching a single JD skill against the resume."""
    jd_skill: str
    requirement_type: str
    status: str              # exact_match | alias_match | related_match | missing
    matched_resume_skill: Optional[str]
    related_resume_skills: List[str]
    resume_evidence: List[Dict]
    explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# SECTION 4 — TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Clean special characters and normalize whitespace."""
    if not text:
        return ""
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2022": "\n",
        "\u00b7": "\n", "\u2192": " ", "\u00a0": " ",
        "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_lookup(text: str) -> str:
    """Lowercase, strip, preserve C++, C#, Node.js etc."""
    text = text.lower().strip()
    text = text.replace("&", " and ").replace("_", " ")
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,-/")


def split_compound_phrase(text: str) -> List[str]:
    """
    Split slash/comma/and-separated technology lists.
    Does not break C++ or Node.js.
    """
    if not text:
        return []
    text = re.sub(r"\s+(?:and|or)\s+", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[,;|]\s*", " | ", text)
    if not re.search(r"https?://", text, flags=re.IGNORECASE):
        text = re.sub(r"\s*/\s*", " | ", text)
    return [
        part.strip(" .:-")
        for part in text.split("|")
        if part.strip(" .:-")
    ]


def is_noise_phrase(phrase: str) -> bool:
    """Return True if the phrase should not be treated as a skill."""
    normalized = normalize_for_lookup(phrase)
    if not normalized:
        return True
    if normalized in NOISE_TERMS:
        return True
    if len(normalized) < 2 or len(normalized) > 70:
        return True
    if normalized.isdigit():
        return True
    if not re.search(r"[a-zA-Z+#]", normalized):
        return True
    if "\n" in phrase:
        return True
    for pattern in ROLE_PATTERNS:
        if re.fullmatch(pattern, normalized, flags=re.IGNORECASE):
            return True
    return False


# ============================================================
# SECTION 5 — SECTION PARSER
# ============================================================

# Regex patterns to detect the start of a resume section heading.
_SECTION_PATTERNS: Dict[str, List[str]] = {
    "Work Experience": [
        r"(?:work|professional|employment)\s*experience",
        r"experience", r"work history", r"career history",
    ],
    "Projects": [
        r"projects?", r"personal projects?", r"academic projects?",
        r"key projects?",
    ],
    "Technical Skills": [
        r"technical\s+skills?", r"skills?\s+(?:&|and)?\s*(?:competencies|expertise)?",
        r"core\s+competencies", r"technologies", r"tech\s+stack",
        r"programming\s+skills?",
    ],
    "Certifications": [
        r"certifications?", r"certificates?", r"credentials?",
        r"professional\s+certifications?",
    ],
    "Summary": [
        r"(?:professional\s+)?summary", r"objective", r"career\s+objective",
        r"profile", r"about\s+me",
    ],
    "Education": [
        r"education(?:al)?\s*(?:background|qualifications?)?",
        r"academic\s+(?:background|qualifications?|history)?",
        r"degrees?", r"qualifications?",
    ],
}

# Build a single compiled pattern per section
_COMPILED_SECTION_RE: Dict[str, re.Pattern] = {}
for _section_name, _patterns in _SECTION_PATTERNS.items():
    combined = "|".join(f"(?:{p})" for p in _patterns)
    _COMPILED_SECTION_RE[_section_name] = re.compile(
        rf"^(?:{combined})\s*[:\-–]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

# Inline heading: "Technical Skills: Python, Docker"
_INLINE_SECTION_RE: Dict[str, re.Pattern] = {}
for _section_name, _patterns in _SECTION_PATTERNS.items():
    combined = "|".join(f"(?:{p})" for p in _patterns)
    _INLINE_SECTION_RE[_section_name] = re.compile(
        rf"^(?:{combined})\s*[:\-–]",
        re.IGNORECASE | re.MULTILINE,
    )


class SectionParser:
    """
    Splits resume text into named sections.
    Returns a list of (section_name, text_chunk) tuples.
    """

    def parse(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect section boundaries and return list of
        (section_name, section_text) ordered by position.
        """
        text = normalize_text(text)
        boundaries: List[Tuple[int, str]] = []  # (char_offset, section_name)

        # Detect standalone heading lines
        for line_match in re.finditer(r"^[^\n]{2,60}$", text, re.MULTILINE):
            line = line_match.group().strip()
            for section_name, pattern in _COMPILED_SECTION_RE.items():
                if pattern.match(line.strip()):
                    boundaries.append((line_match.start(), section_name))
                    break

        # Detect inline headings ("Technical Skills: Python, ...")
        for section_name, pattern in _INLINE_SECTION_RE.items():
            for m in pattern.finditer(text):
                if not any(abs(m.start() - b[0]) < 5 for b in boundaries):
                    boundaries.append((m.start(), section_name))

        if not boundaries:
            return [("Unknown", text)]

        # Sort by position
        boundaries.sort(key=lambda item: item[0])

        sections: List[Tuple[str, str]] = []
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            chunk = text[start:end].strip()
            # Remove the heading line itself
            chunk_lines = chunk.split("\n")
            if chunk_lines:
                chunk_lines = chunk_lines[1:]
            sections.append((name, "\n".join(chunk_lines).strip()))

        # Text before first heading → Unknown
        if boundaries[0][0] > 0:
            pre_text = text[: boundaries[0][0]].strip()
            if pre_text:
                sections.insert(0, ("Unknown", pre_text))

        return sections

    def get_section_for_position(
        self,
        sections: List[Tuple[str, str]],
        target_text: str,
    ) -> str:
        """Return the section name that contains the target_text substring."""
        for section_name, section_text in sections:
            if target_text.lower() in section_text.lower():
                return section_name
        return "Unknown"


# ============================================================
# SECTION 6 — SKILL EXTRACTOR  (section-aware)
# ============================================================

# Module-level singleton cache — loaded once per process lifetime
_EXTRACTOR_CACHE: Dict[str, "HybridSkillExtractor"] = {}


def _get_extractor(
    spacy_model: str = "en_core_web_sm",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.82,
) -> "HybridSkillExtractor":
    """Return a cached extractor — avoids repeated model loading."""
    cache_key = f"{spacy_model}|{embedding_model_name}|{semantic_threshold}"
    if cache_key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[cache_key] = HybridSkillExtractor(
            spacy_model=spacy_model,
            embedding_model_name=embedding_model_name,
            semantic_threshold=semantic_threshold,
        )
    return _EXTRACTOR_CACHE[cache_key]


class HybridSkillExtractor:
    """
    Extracts canonical skills from text using a four-level priority:
      1. Exact canonical name match (PhraseMatcher)
      2. Alias match (PhraseMatcher, case-insensitive)
      3. Multi-word phrase match (compound splitting)
      4. Strict semantic fallback (SentenceTransformer, category-guarded)

    Section awareness is handled by SectionParser; every ExtractedSkill
    carries source_section and evidence_weight.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.82,
    ):
        self.nlp = spacy.load(spacy_model)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.semantic_threshold = semantic_threshold

        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_skills: List[str] = []
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._section_parser = SectionParser()

        self._build_taxonomy()
        self._prepare_embeddings()

    def _build_taxonomy(self) -> None:
        """Build alias lookup and PhraseMatcher patterns from SKILL_TAXONOMY."""
        patterns = []
        for canonical, metadata in SKILL_TAXONOMY.items():
            self.canonical_skills.append(canonical)
            all_names = {canonical, *metadata.get("aliases", [])}
            for name in all_names:
                normalized = normalize_for_lookup(name)
                if normalized:
                    self.alias_to_canonical[normalized] = canonical
                    patterns.append(self.nlp.make_doc(name))
        self.matcher.add("TECH_SKILL", patterns)

    def _prepare_embeddings(self) -> None:
        """Pre-encode taxonomy skill names — called once at init."""
        descriptions = [
            f"{skill}, a technical skill in {SKILL_TAXONOMY[skill]['category']}"
            for skill in self.canonical_skills
        ]
        self.taxonomy_embeddings = self.embedding_model.encode(
            descriptions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _make_skill(
        self,
        canonical: str,
        source_text: str,
        match_type: str,
        confidence: float,
        section: str = "Unknown",
    ) -> ExtractedSkill:
        weight = SECTION_EVIDENCE_WEIGHT.get(section, SECTION_EVIDENCE_WEIGHT["Unknown"])
        return ExtractedSkill(
            skill=canonical,
            category=SKILL_TAXONOMY[canonical]["category"],
            match_type=match_type,
            source_text=source_text.strip(),
            source_section=section,
            confidence=round(float(confidence), 3),
            evidence_weight=weight,
            occurrence_count=1,
            sections_found=[section],
        )

    def _exact_match(self, text: str, section: str) -> List[ExtractedSkill]:
        """Run PhraseMatcher on a text chunk and return matched skills."""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        results: List[ExtractedSkill] = []
        for _, start, end in matches:
            span = doc[start:end]
            normalized = normalize_for_lookup(span.text)
            canonical = self.alias_to_canonical.get(normalized)
            if not canonical:
                continue
            # Determine match type: is it canonical name or alias?
            if normalize_for_lookup(canonical) == normalized:
                mtype = "taxonomy_exact"
            else:
                mtype = "alias_exact"
            results.append(
                self._make_skill(canonical, span.text, mtype, 1.0, section)
            )
        return results

    def _candidate_phrases(self, text: str) -> List[str]:
        """
        Extract candidate phrases for semantic fallback ONLY.
        Uses named entities + noun chunks; filters noise aggressively.
        """
        doc = self.nlp(text)
        candidates: Set[str] = set()

        for ent in doc.ents:
            candidates.update(split_compound_phrase(ent.text))

        for chunk in doc.noun_chunks:
            cleaned = re.sub(
                r"^(?:experience|knowledge|proficiency|familiarity|"
                r"understanding|expertise)\s+(?:with|in|of)\s+",
                "",
                chunk.text.strip(),
                flags=re.IGNORECASE,
            )
            candidates.update(split_compound_phrase(cleaned))

        cleaned = []
        for cand in candidates:
            cand = cand.strip(" .,:;-")
            if not is_noise_phrase(cand):
                cleaned.append(cand)

        return sorted(set(cleaned), key=lambda x: (-len(x), x.lower()))

    def _semantic_match(self, candidate: str, section: str) -> Optional[ExtractedSkill]:
        """
        Semantic fallback — strict.
        Only fires when taxonomy exact and alias matching both fail.
        Applies category guard and tighter threshold for single-word candidates.
        """
        if is_noise_phrase(candidate):
            return None

        normalized = normalize_for_lookup(candidate)

        # One final alias check
        if normalized in self.alias_to_canonical:
            canonical = self.alias_to_canonical[normalized]
            return self._make_skill(canonical, candidate, "alias_exact", 1.0, section)

        is_one_word = len(normalized.split()) == 1

        # Reject single generic words that didn't appear in the taxonomy
        if is_one_word and len(normalized) <= 3:
            return None

        emb = self.embedding_model.encode(
            [f"{candidate}, a technical software skill"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = cosine_similarity(emb, self.taxonomy_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # Strict thresholds
        threshold = self.semantic_threshold
        if is_one_word:
            threshold = max(threshold, 0.85)

        if best_score < threshold:
            return None

        canonical = self.canonical_skills[best_idx]
        return self._make_skill(canonical, candidate, "semantic", best_score, section)

    @staticmethod
    def _merge_duplicate_skills(
        skills: List[ExtractedSkill],
    ) -> List[ExtractedSkill]:
        """
        Keep the highest-confidence evidence per canonical skill.
        Also accumulate occurrence_count and sections_found.
        """
        priority = {"taxonomy_exact": 3, "alias_exact": 2, "semantic": 1}
        unique: Dict[str, ExtractedSkill] = {}

        for skill in skills:
            key = skill.skill.lower()
            if key not in unique:
                unique[key] = skill
            else:
                existing = unique[key]
                # Merge occurrence count and sections
                existing.occurrence_count += 1
                if skill.source_section not in existing.sections_found:
                    existing.sections_found.append(skill.source_section)
                # Upgrade if better match type or evidence weight
                new_rank = (
                    priority.get(skill.match_type, 0),
                    skill.evidence_weight,
                    skill.confidence,
                )
                old_rank = (
                    priority.get(existing.match_type, 0),
                    existing.evidence_weight,
                    existing.confidence,
                )
                if new_rank > old_rank:
                    skill.occurrence_count = existing.occurrence_count
                    skill.sections_found = existing.sections_found
                    unique[key] = skill

        return sorted(
            unique.values(),
            key=lambda s: (s.category.lower(), s.skill.lower()),
        )

    # ----------------------------------------------------------
    # Public extraction
    # ----------------------------------------------------------

    def extract_skills(
        self,
        text: str,
        use_semantic_fallback: bool = True,
        section_aware: bool = True,
    ) -> List[ExtractedSkill]:
        """
        Extract skills from text.

        Args:
            text:                 Raw text (resume or JD).
            use_semantic_fallback: Enable semantic fallback (default True).
            section_aware:        Parse resume sections (default True).

        Returns:
            Deduplicated list of ExtractedSkill sorted by category/name.
        """
        text = normalize_text(text)
        if not text:
            return []

        results: List[ExtractedSkill] = []

        if section_aware:
            sections = self._section_parser.parse(text)
        else:
            sections = [("Unknown", text)]

        for section_name, chunk in sections:
            if not chunk.strip():
                continue
            # Taxonomy + alias matching per section
            results.extend(self._exact_match(chunk, section_name))

        if use_semantic_fallback:
            found_sources = {normalize_for_lookup(s.source_text) for s in results}
            found_canonicals = {s.skill.lower() for s in results}

            # Run semantic on the full text to catch cross-section phrases
            for candidate in self._candidate_phrases(text):
                if normalize_for_lookup(candidate) in found_sources:
                    continue
                # Try to determine which section this candidate came from
                sections_list = self._section_parser.parse(text) if section_aware else [("Unknown", text)]
                section_name = "Unknown"
                for sname, schunk in sections_list:
                    if candidate.lower() in schunk.lower():
                        section_name = sname
                        break
                sem = self._semantic_match(candidate, section_name)
                if sem and sem.skill.lower() not in found_canonicals:
                    results.append(sem)
                    found_canonicals.add(sem.skill.lower())

        return self._merge_duplicate_skills(results)


# ============================================================
# SECTION 7 — JD REQUIREMENT CLASSIFIER
# ============================================================

class JDRequirementClassifier:
    """
    Classifies every skill in a JD as mandatory, preferred, or optional
    by scanning a fixed-size context window around the skill mention.

    Default: mandatory (most JD skills are implicitly required).
    """

    WINDOW_CHARS = 160  # characters to look around each skill mention

    def classify(
        self,
        jd_text: str,
        jd_skills: List[ExtractedSkill],
    ) -> List[JDSkill]:
        """
        For each extracted JD skill, determine requirement type by
        looking at nearby text.
        """
        text_lower = jd_text.lower()
        result: List[JDSkill] = []

        for skill in jd_skills:
            req_type = self._classify_single(text_lower, skill.skill)
            result.append(
                JDSkill(
                    skill=skill.skill,
                    category=skill.category,
                    requirement_type=req_type,
                    importance_weight=REQUIREMENT_WEIGHTS[req_type],
                    source_text=skill.source_text,
                )
            )

        return result

    def _classify_single(self, text_lower: str, skill_name: str) -> str:
        """
        Find all occurrences of the skill in the JD and check the window
        around each for requirement keywords.
        """
        skill_lower = skill_name.lower()
        found_positions = [
            m.start()
            for m in re.finditer(re.escape(skill_lower), text_lower)
        ]

        if not found_positions:
            # Try aliases
            aliases = SKILL_TAXONOMY.get(skill_name, {}).get("aliases", [])
            for alias in aliases:
                for m in re.finditer(re.escape(alias.lower()), text_lower):
                    found_positions.append(m.start())
            if not found_positions:
                return "mandatory"  # default

        best = "mandatory"
        # If any window says optional → optional; if preferred → preferred
        # mandatory stays only when no keyword found (default)
        classification_order = ["mandatory", "preferred", "optional"]

        for pos in found_positions:
            start = max(0, pos - self.WINDOW_CHARS)
            end = min(len(text_lower), pos + self.WINDOW_CHARS)
            window = text_lower[start:end]

            for kw in _OPTIONAL_KEYWORDS:
                if kw in window:
                    best = "optional"
                    break
            if best == "optional":
                continue

            for kw in _PREFERRED_KEYWORDS:
                if kw in window:
                    if classification_order.index(best) > classification_order.index("preferred"):
                        best = "preferred"
                    break

        return best


# ============================================================
# SECTION 8 — MATCHER
# ============================================================

class SkillMatcher:
    """
    Matches JD skills against resume skills using a strict priority pipeline:
      1. Exact canonical match
      2. Alias match (via alias_to_canonical table)
      3. Taxonomy related-skill match (informational only, NOT full match)
      4. Semantic similarity (within compatible categories, strict threshold)

    Related and semantic matches are NEVER counted as full matches.
    """

    RELATED_SEMANTIC_THRESHOLD = 0.88  # higher than extraction threshold

    def __init__(self, extractor: HybridSkillExtractor):
        self.extractor = extractor

    def match(
        self,
        resume_skills: List[ExtractedSkill],
        jd_skills: List[JDSkill],
    ) -> List[SkillMatchResult]:
        """
        For every JD skill, determine status and build a SkillMatchResult.
        """
        # Build resume lookup structures
        resume_canonical_lower: Dict[str, ExtractedSkill] = {
            s.skill.lower(): s for s in resume_skills
        }
        resume_alias_lower: Dict[str, ExtractedSkill] = {}
        for rs in resume_skills:
            nk = normalize_for_lookup(rs.skill)
            resume_alias_lower[nk] = rs
            for alias in SKILL_TAXONOMY.get(rs.skill, {}).get("aliases", []):
                resume_alias_lower[normalize_for_lookup(alias)] = rs

        results: List[SkillMatchResult] = []

        for jd_skill in jd_skills:
            result = self._match_single(
                jd_skill=jd_skill,
                resume_skills=resume_skills,
                resume_canonical_lower=resume_canonical_lower,
                resume_alias_lower=resume_alias_lower,
            )
            results.append(result)

        return results

    def _match_single(
        self,
        jd_skill: JDSkill,
        resume_skills: List[ExtractedSkill],
        resume_canonical_lower: Dict[str, ExtractedSkill],
        resume_alias_lower: Dict[str, ExtractedSkill],
    ) -> SkillMatchResult:
        jd_name_lower = jd_skill.skill.lower()
        jd_normalized = normalize_for_lookup(jd_skill.skill)

        # --- Level 1: Exact canonical match ---
        if jd_name_lower in resume_canonical_lower:
            matched = resume_canonical_lower[jd_name_lower]
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="exact_match",
                matched_resume_skill=matched.skill,
                related_resume_skills=[],
                resume_evidence=[matched.to_dict()],
                explanation=f"Exact match found: '{matched.skill}' in resume.",
            )

        # --- Level 2: Alias match ---
        if jd_normalized in resume_alias_lower:
            matched = resume_alias_lower[jd_normalized]
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="alias_match",
                matched_resume_skill=matched.skill,
                related_resume_skills=[],
                resume_evidence=[matched.to_dict()],
                explanation=(
                    f"Alias match: '{matched.skill}' in resume maps to "
                    f"'{jd_skill.skill}' via alias."
                ),
            )
        # Also check JD skill's aliases against resume canonicals
        for alias in SKILL_TAXONOMY.get(jd_skill.skill, {}).get("aliases", []):
            alias_norm = normalize_for_lookup(alias)
            if alias_norm in resume_alias_lower:
                matched = resume_alias_lower[alias_norm]
                return SkillMatchResult(
                    jd_skill=jd_skill.skill,
                    requirement_type=jd_skill.requirement_type,
                    status="alias_match",
                    matched_resume_skill=matched.skill,
                    related_resume_skills=[],
                    resume_evidence=[matched.to_dict()],
                    explanation=(
                        f"Alias match: '{matched.skill}' in resume satisfies "
                        f"'{jd_skill.skill}' (alias: '{alias}')."
                    ),
                )

        # --- Level 3: Taxonomy related-skill identification ---
        # (informational — does NOT count as a match)
        related_in_taxonomy = set(
            SKILL_TAXONOMY.get(jd_skill.skill, {}).get("related_skills", [])
        )
        # Also check narrower_skills and broader_skill
        broader = SKILL_TAXONOMY.get(jd_skill.skill, {}).get("broader_skill", "")
        narrower = set(SKILL_TAXONOMY.get(jd_skill.skill, {}).get("narrower_skills", []))
        all_related = related_in_taxonomy | narrower
        if broader:
            all_related.add(broader)

        related_found_in_resume = [
            rs.skill
            for rs in resume_skills
            if rs.skill in all_related
        ]

        # --- Level 4: Semantic related detection (still NOT a full match) ---
        jd_category = jd_skill.category
        semantic_related: List[str] = []

        if resume_skills:
            jd_emb = self.extractor.embedding_model.encode(
                [f"{jd_skill.skill}, a technical skill in {jd_category}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            resume_descs = [
                f"{rs.skill}, a technical skill in {rs.category}"
                for rs in resume_skills
            ]
            res_embs = self.extractor.embedding_model.encode(
                resume_descs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sims = cosine_similarity(jd_emb, res_embs)[0]
            for idx, score in enumerate(sims):
                rs = resume_skills[idx]
                # Same or compatible category guard
                same_cat = rs.category == jd_category
                compatible = self._compatible_categories(rs.category, jd_category)
                if (same_cat or compatible) and score >= self.RELATED_SEMANTIC_THRESHOLD:
                    if rs.skill not in related_found_in_resume:
                        semantic_related.append(rs.skill)

        all_related_resume = list(
            dict.fromkeys(related_found_in_resume + semantic_related)
        )

        if all_related_resume:
            return SkillMatchResult(
                jd_skill=jd_skill.skill,
                requirement_type=jd_skill.requirement_type,
                status="related_match",
                matched_resume_skill=None,
                related_resume_skills=all_related_resume,
                resume_evidence=[],
                explanation=(
                    f"Related concepts found ({', '.join(all_related_resume)}), "
                    f"but explicit '{jd_skill.skill}' evidence was not found in the resume."
                ),
            )

        # --- Missing ---
        return SkillMatchResult(
            jd_skill=jd_skill.skill,
            requirement_type=jd_skill.requirement_type,
            status="missing",
            matched_resume_skill=None,
            related_resume_skills=[],
            resume_evidence=[],
            explanation=f"'{jd_skill.skill}' was not found in the resume.",
        )

    @staticmethod
    def _compatible_categories(cat_a: str, cat_b: str) -> bool:
        """Return True if two categories are considered semantically compatible."""
        groups = [
            {"Deep Learning Framework", "AI/ML", "ML Framework", "NLP Framework",
             "Computer Vision Library", "NLP Library"},
            {"Backend Framework", "Backend Runtime", "Backend"},
            {"Database", "Query Language"},
            {"Frontend Framework", "Frontend Library", "Frontend"},
            {"Cloud", "DevOps"},
            {"Data Science Library", "Data Science", "Data Visualization", "Scientific Computing"},
            {"Programming Language", "Scripting"},
        ]
        for group in groups:
            if cat_a in group and cat_b in group:
                return True
        return False


# ============================================================
# SECTION 9 — WEIGHTED SCORER
# ============================================================

class WeightedScorer:
    """
    Computes a composite ATS-style weighted score from match results.

    Score components:
      - mandatory_skill_coverage (50%)
      - preferred_skill_coverage (20%)
      - evidence_strength        (15%)
      - project_experience_relevance (10%)
      - education_or_certification_relevance (5%)
    """

    def score(
        self,
        match_results: List[SkillMatchResult],
        resume_skills: List[ExtractedSkill],
    ) -> Dict:
        mandatory = [r for r in match_results if r.requirement_type == "mandatory"]
        preferred = [r for r in match_results if r.requirement_type == "preferred"]
        optional  = [r for r in match_results if r.requirement_type == "optional"]

        def _coverage(results: List[SkillMatchResult]) -> float:
            if not results:
                return 0.0
            total = sum(MATCH_SCORES.get(r.status, 0.0) for r in results)
            return total / len(results)

        mandatory_cov = _coverage(mandatory)
        preferred_cov = _coverage(preferred)

        # Evidence strength: average of best evidence weights for matched skills
        matched_evidence_weights = [
            e["evidence_weight"]
            for r in match_results
            if r.status in ("exact_match", "alias_match")
            for e in r.resume_evidence
        ]
        evidence_strength = (
            float(np.mean(matched_evidence_weights)) if matched_evidence_weights else 0.0
        )

        # Project/experience relevance: fraction of matched skills seen in Work Experience or Projects
        project_exp_sections = {"Work Experience", "Projects"}
        project_exp_skills = [
            rs
            for rs in resume_skills
            if any(s in project_exp_sections for s in rs.sections_found)
        ]
        project_exp_relevance = min(1.0, len(project_exp_skills) / max(1, len(resume_skills)))

        # Education/certification relevance
        edu_cert_sections = {"Education", "Certifications"}
        edu_cert_skills = [
            rs
            for rs in resume_skills
            if any(s in edu_cert_sections for s in rs.sections_found)
        ]
        edu_cert_relevance = min(1.0, len(edu_cert_skills) / max(1, len(resume_skills)))

        # Composite weighted score
        overall = (
            FINAL_SCORE_WEIGHTS["mandatory_skill_coverage"] * mandatory_cov
            + FINAL_SCORE_WEIGHTS["preferred_skill_coverage"] * preferred_cov
            + FINAL_SCORE_WEIGHTS["evidence_strength"] * evidence_strength
            + FINAL_SCORE_WEIGHTS["project_experience_relevance"] * project_exp_relevance
            + FINAL_SCORE_WEIGHTS["education_or_certification_relevance"] * edu_cert_relevance
        )

        exact_count = sum(1 for r in match_results if r.status == "exact_match")
        alias_count = sum(1 for r in match_results if r.status == "alias_match")
        related_count = sum(1 for r in match_results if r.status == "related_match")

        return {
            "mandatory_skill_coverage": round(mandatory_cov, 4),
            "preferred_skill_coverage": round(preferred_cov, 4),
            "evidence_strength": round(evidence_strength, 4),
            "project_experience_relevance": round(project_exp_relevance, 4),
            "education_or_certification_relevance": round(edu_cert_relevance, 4),
            "overall_skill_score": round(overall, 4),
            "exact_match_count": exact_count + alias_count,
            "alias_match_count": alias_count,
            "related_match_count": related_count,
            "missing_mandatory_count": sum(
                1 for r in mandatory if r.status in ("missing", "related_match")
            ),
            "missing_preferred_count": sum(
                1 for r in preferred if r.status in ("missing", "related_match")
            ),
        }


# ============================================================
# SECTION 10 — SKILL GAP ANALYZER (orchestrator)
# ============================================================

class SkillGapAnalyzer:
    """
    Orchestrates the full skill gap analysis pipeline.

    Usage:
        analyzer = SkillGapAnalyzer()
        result = analyzer.compare(resume_text, jd_text)
    """

    def __init__(
        self,
        extractor: Optional[HybridSkillExtractor] = None,
        semantic_threshold: float = 0.82,
    ):
        self.extractor = extractor or _get_extractor(
            semantic_threshold=semantic_threshold
        )
        self.classifier = JDRequirementClassifier()
        self.matcher = SkillMatcher(self.extractor)
        self.scorer = WeightedScorer()

    def compare(
        self,
        resume_text: str,
        jd_text: str,
        allow_related_skill_match: bool = False,  # kept for backward compat; not used
    ) -> Dict:
        """
        Full pipeline: extract → classify → match → score → return.

        Returns a structured dict with:
            resume_skills, jd_required_skills, matched_skills,
            partially_related_skills, missing_mandatory_skills,
            missing_preferred_skills, additional_resume_skills,
            missing_skills (backward-compat union),
            skill_analysis, summary
        """
        # 1. Extract resume skills (section-aware)
        resume_skills: List[ExtractedSkill] = self.extractor.extract_skills(
            resume_text,
            use_semantic_fallback=True,
            section_aware=True,
        )

        # 2. Extract JD skills (no section parsing needed for JD)
        jd_raw_skills: List[ExtractedSkill] = self.extractor.extract_skills(
            jd_text,
            use_semantic_fallback=False,
            section_aware=False,
        )

        # 3. Classify JD skills by requirement type
        jd_skills: List[JDSkill] = self.classifier.classify(jd_text, jd_raw_skills)

        # 4. Match
        match_results: List[SkillMatchResult] = self.matcher.match(
            resume_skills=resume_skills,
            jd_skills=jd_skills,
        )

        # 5. Score
        summary = self.scorer.score(match_results, resume_skills)

        # 6. Build output lists
        matched = [r for r in match_results if r.status in ("exact_match", "alias_match")]
        partially_related = [r for r in match_results if r.status == "related_match"]
        missing = [r for r in match_results if r.status in ("missing", "related_match")]

        missing_mandatory = [
            r.jd_skill for r in match_results
            if r.requirement_type == "mandatory"
            and r.status in ("missing", "related_match")
        ]
        missing_preferred = [
            r.jd_skill for r in match_results
            if r.requirement_type == "preferred"
            and r.status in ("missing", "related_match")
        ]

        resume_skill_names = {rs.skill for rs in resume_skills}
        jd_skill_names = {js.skill for js in jd_skills}
        additional_resume_skills = sorted(resume_skill_names - jd_skill_names)

        return {
            "resume_skills": [rs.to_dict() for rs in resume_skills],
            "jd_required_skills": [js.to_dict() for js in jd_skills],
            "matched_skills": [r.to_dict() for r in matched],
            "partially_related_skills": [r.to_dict() for r in partially_related],
            "missing_mandatory_skills": sorted(missing_mandatory),
            "missing_preferred_skills": sorted(missing_preferred),
            "additional_resume_skills": additional_resume_skills,
            # backward-compat: union of mandatory + preferred missing skills
            "missing_skills": sorted(set(missing_mandatory + missing_preferred)),
            "skill_analysis": [r.to_dict() for r in match_results],
            "summary": summary,
        }


# ============================================================
# SECTION 11 — PUBLIC API  (backward-compatible)
# ============================================================

# Module-level singletons (loaded once when the module is imported)
_default_extractor: Optional[HybridSkillExtractor] = None
_default_analyzer: Optional[SkillGapAnalyzer] = None


def _ensure_defaults() -> None:
    """Lazy-initialize module-level singletons on first call."""
    global _default_extractor, _default_analyzer
    if _default_extractor is None:
        _default_extractor = _get_extractor(semantic_threshold=0.82)
    if _default_analyzer is None:
        _default_analyzer = SkillGapAnalyzer(extractor=_default_extractor)


def extract_skills(text: str) -> List[Dict]:
    """
    Extract canonical skills from text.

    Args:
        text: Resume or job description text.

    Returns:
        List of skill dicts (see ExtractedSkill.to_dict()).
    """
    _ensure_defaults()
    skills = _default_extractor.extract_skills(text)
    return [s.to_dict() for s in skills]


def find_missing_skills(
    resume_text: str,
    jd_text: str,
    allow_related_skill_match: bool = False,
) -> Dict:
    """
    Full hybrid ATS-grade skill gap analysis.

    Args:
        resume_text:              Raw resume text (plain text or extracted PDF text).
        jd_text:                  Raw job description text.
        allow_related_skill_match: Kept for backward compatibility; not used.

    Returns:
        Dict with keys:
            resume_skills, jd_required_skills, matched_skills,
            partially_related_skills, missing_mandatory_skills,
            missing_preferred_skills, additional_resume_skills,
            missing_skills, skill_analysis, summary.
    """
    _ensure_defaults()
    return _default_analyzer.compare(
        resume_text=resume_text,
        jd_text=jd_text,
        allow_related_skill_match=allow_related_skill_match,
    )


# ============================================================
# QUICK SELF-TEST  (run directly with: python sementic_meaning.py)
# ============================================================

if __name__ == "__main__":
    from pprint import pprint

    sample_resume = """
    TECHNICAL SKILLS
    Languages: Python, Java and C++
    Machine Learning: Machine Learning, Neural Networks, scikit-learn and Pandas
    Databases: MySQL, PostgreSQL and DBMS
    Backend: FastAPI, REST API
    Tools: Git, GitHub and Docker

    WORK EXPERIENCE
    Software Developer Intern — ABC Corp (2024)
    - Built REST APIs using FastAPI and Python
    - Worked with Docker for containerization
    - Used PostgreSQL for data storage

    PROJECTS
    - Sentiment Analysis using NLTK and scikit-learn
    - Image classification with CNN (TensorFlow)

    EDUCATION
    B.E. Computer Science, XYZ University, 2024
    """

    sample_jd = """
    We are looking for a Machine Learning Engineer.

    Requirements:
    - Must have strong Python programming experience
    - Required: TensorFlow or PyTorch
    - Experience with scikit-learn is required
    - Docker is mandatory

    Preferred:
    - Nice to have experience with Kubernetes
    - Preferred: knowledge of MLflow

    Optional:
    - Exposure to Apache Airflow
    - Familiarity with GraphQL
    """

    result = find_missing_skills(resume_text=sample_resume, jd_text=sample_jd)

    print("\n=== RESUME SKILLS ===")
    for s in result["resume_skills"]:
        print(f"  {s['skill']:35s} [{s['source_section']}]  ev={s['evidence_weight']}")

    print("\n=== JD REQUIRED SKILLS ===")
    for s in result["jd_required_skills"]:
        print(f"  {s['skill']:35s} [{s['requirement_type']}]")

    print("\n=== SKILL ANALYSIS ===")
    for entry in result["skill_analysis"]:
        status = entry["status"]
        mark = "✓" if status in ("exact_match", "alias_match") else "~" if status == "related_match" else "✗"
        related = f"  related={entry['related_resume_skills']}" if entry["related_resume_skills"] else ""
        print(f"  {mark} [{entry['requirement_type']:9s}] {entry['jd_skill']:30s} → {status}{related}")

    print("\n=== MISSING MANDATORY ===", result["missing_mandatory_skills"])
    print("=== MISSING PREFERRED  ===", result["missing_preferred_skills"])
    print("\n=== SUMMARY ===")
    pprint(result["summary"])