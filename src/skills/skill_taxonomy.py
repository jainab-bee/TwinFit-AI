from __future__ import annotations

from typing import Dict


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
