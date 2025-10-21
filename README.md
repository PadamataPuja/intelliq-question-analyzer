      IntelliQ – Intelligent Stack Overflow Question Quality Analyzer

IntelliQ is an AI-powered system that combines deep learning and NLP techniques to evaluate and improve the quality of technical questions, inspired by Stack Overflow's curation standards.

The project consists of two primary components:

BERT-based Question Quality Classifier

Fine-tuned using the bert-base-uncased model.

Categorizes questions into one of three classes:

High Quality (HQ)

Low Quality – Needs Edit (LQ_EDIT)

Very Low Quality – Needs Close (LQ_CLOSE)

Suggestion & Improvement Engine

Leverages semantic similarity techniques and rule-based analysis.

Provides contextual recommendations, including:

Adding error details or tracebacks

Improving question structure and title clarity

Supplying reproducible code snippets

Clarifying the technical context and environment

⚙️ Features
State-of-the-art BERT classifier tailored for Stack Overflow-style question quality assessment.

Semantic search using SBERT to retrieve comparable high-quality questions for user guidance.

Automated text quality analysis for readability and structural metrics.

End-to-end data preprocessing, label encoding, and evaluation pipeline.

Generates natural language feedback to assist users in improving their questions.

🧠 Model Workflow
Preprocess training (train.csv) and validation (valid.csv) datasets.

Fine-tune the BERT classification model using model.sbert.py.

Save the best model weights as bert_stackoverflow_model.pth.

Use suggestion_model.py to:

Predict question quality,

Perform pattern-based textual analysis,

Generate actionable improvement suggestions.

🧩 Example Usage
Train the model
Run:

bash
python model.sbert.py
Get suggestions for questions
Run:

bash
python suggestion_model.py
Then enter the question title and body when prompted to receive quality predictions and improvement tips along with similar well-written example questions.

📦 Outputs
accuracy_loss.png: Graphs displaying training and validation accuracy and loss.

confusion_matrix.png: Confusion matrix overview across classes.

bert_stackoverflow_model.pth: Saved fine-tuned BERT model file.
