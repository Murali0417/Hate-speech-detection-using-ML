# Hate-speech-detection-using-ML
📝 Project Description
This project uses Python and machine learning to detect hate speech in tweets. A labeled Kaggle dataset is preprocessed and vectorized using TF-IDF. Models like SVM and Naive Bayes are trained, with SVM performing best. It helps identify offensive content for safer online communication.
Hate Speech Detection using Machine Learning

This project is an end-to-end machine learning pipeline for detecting hate speech in text data. It uses a classical machine learning approach, combining a TF-IDF vectorizer for feature extraction with a Logistic Regression model for classification. The entire workflow, from data preprocessing and exploratory data analysis (EDA) to model training and hyperparameter tuning, is contained within a single script.

The primary goal is to accurately classify text as either hate speech or non-hate speech, providing a solid baseline for content moderation tasks.

✨ Features

    Automated Setup: Automatically downloads necessary NLTK resources (stopwords, punkt, wordnet).

    Robust Text Preprocessing: Cleans and normalizes raw text data by removing URLs, mentions, and special characters, followed by stop word removal and lemmatization.

    Exploratory Data Analysis (EDA): Visualizes the data distribution with count plots, pie charts, and word clouds to gain insights into the dataset.

    Efficient Feature Extraction: Uses TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams to convert text into a numerical format suitable for machine learning.

    Model Training & Evaluation: Trains a Logistic Regression model and evaluates its performance using key metrics like accuracy, precision, recall, and F1-score.

    Hyperparameter Tuning: Optimizes the model's performance using Grid Search with Cross-Validation to find the best parameters.

⚙️ Technologies & Libraries

    pandas: For data manipulation and loading.

    scikit-learn: For machine learning model building, feature extraction, and evaluation.

    nltk: For advanced text preprocessing (tokenization, stop words, lemmatization).

    matplotlib & seaborn: For data visualization.

    re: For regular expression-based text cleaning.

    wordcloud: To visualize the most frequent words in the dataset.

🚀 Installation

    Clone the repository:

    Install dependencies:

    Note: Your script automatically handles the download of NLTK data. A simple pip install nltk is sufficient.

📂 Data

This script requires a CSV file named hateDetection_train.csv to be present in the same directory. The file should contain at least two columns:

    tweet: The text content to be analyzed.

    label: The corresponding label (e.g., 0 for non-hate speech, 1 for hate speech).

📊 Workflow & Usage

Simply run the script from your terminal. It will execute the entire pipeline from start to finish, printing the results to the console.

The script will perform the following steps:

    Load and preprocess the data.

    Generate visualizations (count plot, pie chart, two word clouds). These plots will be displayed automatically.

    Vectorize the tweets using TfidfVectorizer.

    Train and evaluate the initial LogisticRegression model, printing the accuracy, confusion matrix, and classification report.

    Perform hyperparameter tuning with GridSearchCV.

    Re-evaluate the model with the optimized parameters and print the final performance metrics.

📈 Expected Output

When you run the script, you will see a series of plots and console outputs, including:

    Plots showing the data distribution and word clouds.

    The number of features created by the vectorizer.

    The size of the train and test sets.

    The initial test accuracy and classification report for the baseline model.

    The confusion matrix plot for the baseline model.

    The best cross-validation score and best parameters found by GridSearchCV.

    The final test accuracy and classification report for the optimized model.

🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests to enhance the project.
