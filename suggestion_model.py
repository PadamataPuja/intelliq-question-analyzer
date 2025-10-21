import torch
import numpy as np
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configuration
MODEL_PATH = r"bert_stackoverflow_model.pth"
TRAIN_DATASET_PATH = r"dataset\train.csv"
TEST_DATASET_PATH = r"dataset\valid.csv"
NUM_LABELS = 3
MAX_LEN = 128

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Running on CPU - Optimizing for multi-core")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

# Load Model & Tokenizer
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Define LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["LQ_CLOSE", "LQ_EDIT", "HQ"])

# Text Preprocessing
def clean_html(text):
    return re.sub("<.*?>", "", str(text))

def preprocess_text(text):
    """Enhanced text preprocessing for better IR"""
    text = clean_html(str(text))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Question Quality Analyzer
class QuestionQualityAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.hq_patterns = None
        self.common_tech_terms = set()
        
    def analyze_hq_patterns(self, hq_questions):
        """Analyze patterns in high-quality questions"""
        patterns = {
            'avg_title_length': [],
            'avg_body_length': [],
            'avg_sentences': [],
            'common_title_starters': Counter(),
            'common_keywords': Counter(),
            'has_code_examples': 0,
            'has_error_messages': 0,
            'has_specific_versions': 0,
            'question_types': Counter(),
            'technical_terms': Counter()
        }
        
        for question in hq_questions:
            title = question.get('title', '')
            body = question.get('body', '')
            combined = title + " " + body
            
            # Basic metrics
            patterns['avg_title_length'].append(len(title.split()))
            patterns['avg_body_length'].append(len(body.split()))
            patterns['avg_sentences'].append(len(sent_tokenize(body)))
            
            # Title analysis
            title_lower = title.lower()
            first_word = title_lower.split()[0] if title_lower.split() else ""
            patterns['common_title_starters'][first_word] += 1
            
            # Question type detection
            if any(word in title_lower for word in ['how to', 'how do', 'how can']):
                patterns['question_types']['how-to'] += 1
            elif any(word in title_lower for word in ['what is', 'what does', 'what are']):
                patterns['question_types']['what'] += 1
            elif any(word in title_lower for word in ['why does', 'why is', 'why are']):
                patterns['question_types']['why'] += 1
            elif '?' in title:
                patterns['question_types']['direct-question'] += 1
            else:
                patterns['question_types']['statement'] += 1
            
            # Content analysis
            combined_lower = combined.lower()
            
            # Check for code examples
            if any(indicator in combined_lower for indicator in ['```', 'code', 'function', 'class', 'def ', 'import ', 'from ']):
                patterns['has_code_examples'] += 1
            
            # Check for error messages
            if any(indicator in combined_lower for indicator in ['error', 'exception', 'traceback', 'failed', 'cannot', "doesn't work"]):
                patterns['has_error_messages'] += 1
            
            # Check for specific versions
            if re.search(r'\d+\.\d+', combined):
                patterns['has_specific_versions'] += 1
            
            # Extract technical terms and keywords
            words = word_tokenize(combined_lower)
            words = [word for word in words if word.isalnum() and word not in self.stop_words and len(word) > 2]
            
            for word in words:
                patterns['common_keywords'][word] += 1
                # Identify technical terms (words with specific patterns)
                if any(tech in word for tech in ['api', 'http', 'json', 'xml', 'sql', 'html', 'css', 'js']):
                    patterns['technical_terms'][word] += 1
        
        # Calculate averages
        total_questions = len(hq_questions)
        if total_questions > 0:
            patterns['avg_title_length'] = np.mean(patterns['avg_title_length'])
            patterns['avg_body_length'] = np.mean(patterns['avg_body_length'])
            patterns['avg_sentences'] = np.mean(patterns['avg_sentences'])
            patterns['has_code_examples'] = patterns['has_code_examples'] / total_questions
            patterns['has_error_messages'] = patterns['has_error_messages'] / total_questions
            patterns['has_specific_versions'] = patterns['has_specific_versions'] / total_questions
        
        # Fix: Extract only the terms from most_common, not the (term, count) tuples
        self.common_tech_terms = {term for term, count in patterns['technical_terms'].most_common(100)}
        
        self.hq_patterns = patterns
        return patterns
    
    def analyze_question_quality(self, title, body):
        """Analyze a question and provide detailed feedback"""
        if not self.hq_patterns:
            return {"error": "HQ patterns not analyzed yet"}
        
        combined = title + " " + body
        analysis = {
            'title_analysis': self._analyze_title(title),
            'body_analysis': self._analyze_body(body),
            'content_analysis': self._analyze_content(combined),
            'suggestions': []
        }
        
        # Generate suggestions based on analysis
        analysis['suggestions'] = self._generate_suggestions(analysis, title, body)
        
        return analysis
    
    def _analyze_title(self, title):
        """Analyze title quality"""
        words = title.split()
        analysis = {
            'length': len(words),
            'is_question': '?' in title,
            'starts_with_question_word': False,
            'specificity_score': 0,
            'has_tech_terms': False
        }
        
        # Check if starts with question words
        title_lower = title.lower()
        question_starters = ['how', 'what', 'why', 'when', 'where', 'which', 'can', 'should', 'is', 'are', 'do', 'does']
        analysis['starts_with_question_word'] = any(title_lower.startswith(word) for word in question_starters)
        
        # Check for technical terms
        analysis['has_tech_terms'] = any(term in title_lower for term in self.common_tech_terms)
        
        # Specificity score (based on presence of specific terms, versions, etc.)
        specific_indicators = len(re.findall(r'\d+\.\d+|\b[A-Z]{2,}\b|api|http|json|error|exception', title_lower))
        analysis['specificity_score'] = min(specific_indicators / 3, 1.0)  # Normalize to 0-1
        
        return analysis
    
    def _analyze_body(self, body):
        """Analyze body quality"""
        sentences = sent_tokenize(body)
        words = body.split()
        
        analysis = {
            'length': len(words),
            'sentence_count': len(sentences),
            'has_code': False,
            'has_error_details': False,
            'has_context': False,
            'structure_score': 0
        }
        
        body_lower = body.lower()
        
        # Check for code examples
        code_indicators = ['```', 'code', 'function', 'def ', 'class ', 'import ', '<code>', 'script']
        analysis['has_code'] = any(indicator in body_lower for indicator in code_indicators)
        
        # Check for error details
        error_indicators = ['error', 'traceback', 'exception', 'failed', 'returns', 'expected', 'actual']
        analysis['has_error_details'] = any(indicator in body_lower for indicator in error_indicators)
        
        # Check for context
        context_indicators = ['trying to', 'want to', 'need to', 'using', 'working with', 'version', 'environment']
        analysis['has_context'] = any(indicator in body_lower for indicator in context_indicators)
        
        # Structure score (based on sentence count and organization)
        if len(sentences) >= 3:
            analysis['structure_score'] = 1.0
        elif len(sentences) >= 2:
            analysis['structure_score'] = 0.7
        else:
            analysis['structure_score'] = 0.3
        
        return analysis
    
    def _analyze_content(self, combined_text):
        """Analyze overall content quality"""
        analysis = {
            'readability_score': 0,
            'technical_depth': 0,
            'clarity_score': 0
        }
        
        words = word_tokenize(combined_text.lower())
        words = [word for word in words if word.isalnum()]
        
        # Technical depth (based on technical terms)
        tech_term_count = sum(1 for word in words if word in self.common_tech_terms)
        analysis['technical_depth'] = min(tech_term_count / max(len(words) * 0.1, 1), 1.0)
        
        # Clarity score (based on sentence structure and common words)
        sentences = sent_tokenize(combined_text)
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Ideal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            analysis['clarity_score'] = 1.0
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            analysis['clarity_score'] = 0.7
        else:
            analysis['clarity_score'] = 0.4
        
        return analysis
    
    def _generate_suggestions(self, analysis, title, body):
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Title suggestions
        title_analysis = analysis['title_analysis']
        if title_analysis['length'] < 5:
            suggestions.append({
                'type': 'title',
                'priority': 'high',
                'message': 'Your title is too short. Add more specific details about your problem.',
                'example': 'Instead of "API problem", try "Cannot connect to REST API - getting timeout error with Python requests"'
            })
        
        if not title_analysis['starts_with_question_word'] and not title_analysis['is_question']:
            suggestions.append({
                'type': 'title',
                'priority': 'medium',
                'message': 'Consider starting your title with a question word (How, What, Why) or ending with a question mark.',
                'example': 'Instead of "API connection issue", try "How to fix API connection timeout in Python?"'
            })
        
        if title_analysis['specificity_score'] < 0.3:
            suggestions.append({
                'type': 'title',
                'priority': 'high',
                'message': 'Make your title more specific. Include technology names, error types, or version numbers.',
                'example': 'Add details like "Python 3.9", "REST API", "timeout error", or specific library names'
            })
        
        # Body suggestions
        body_analysis = analysis['body_analysis']
        if body_analysis['length'] < 20:
            suggestions.append({
                'type': 'body',
                'priority': 'high',
                'message': 'Your question body is too short. Provide more context and details.',
                'example': 'Explain what you\'re trying to achieve, what you\'ve tried, and what specific error you\'re getting'
            })
        
        if not body_analysis['has_code']:
            suggestions.append({
                'type': 'body',
                'priority': 'medium',
                'message': 'Include relevant code examples to help others understand your problem.',
                'example': 'Show the code that\'s causing the issue, even if it\'s just a few lines'
            })
        
        if not body_analysis['has_error_details']:
            suggestions.append({
                'type': 'body',
                'priority': 'high',
                'message': 'Include specific error messages, unexpected behavior, or what you expected vs. what you got.',
                'example': 'Copy the exact error message, or describe precisely what happens vs. what should happen'
            })
        
        if not body_analysis['has_context']:
            suggestions.append({
                'type': 'body',
                'priority': 'medium',
                'message': 'Provide more context about your environment, what you\'re trying to accomplish, and what you\'ve already tried.',
                'example': 'Mention your programming language version, libraries used, and any troubleshooting steps you\'ve taken'
            })
        
        if body_analysis['sentence_count'] < 2:
            suggestions.append({
                'type': 'structure',
                'priority': 'medium',
                'message': 'Structure your question better with multiple sentences or paragraphs.',
                'example': 'Use separate sentences for: 1) What you\'re trying to do, 2) What you\'ve tried, 3) What error you\'re getting'
            })
        
        # Content suggestions
        content_analysis = analysis['content_analysis']
        if content_analysis['technical_depth'] < 0.2:
            suggestions.append({
                'type': 'content',
                'priority': 'medium',
                'message': 'Include more technical details relevant to your problem.',
                'example': 'Mention specific technologies, libraries, frameworks, or tools you\'re using'
            })
        
        return suggestions

# Enhanced Dataset Loader with Pattern Analysis
class EnhancedHQDatasetLoader:
    def __init__(self, train_dataset_path, test_dataset_path):
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.hq_questions = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_embeddings = None
        self.quality_analyzer = QuestionQualityAnalyzer()
        
    def load_hq_questions(self):
        """Load HQ questions from both datasets"""
        try:
            all_hq_questions = []
            
            # Define label mappings (handle both string and numeric labels)
            label_mappings = {
                'HQ': ['HQ', 2],  # HQ can be string 'HQ' or numeric 2
                'LQ_EDIT': ['LQ_EDIT', 1],
                'LQ_CLOSE': ['LQ_CLOSE', 0]
            }
            
            # Load training dataset
            if os.path.exists(self.train_dataset_path):
                train_df = pd.read_csv(self.train_dataset_path)
                print(f"Training dataset shape: {train_df.shape}")
                print(f"Training dataset columns: {train_df.columns.tolist()}")
                
                # Handle different column naming conventions
                if 'Y' in train_df.columns:
                    label_col = 'Y'
                elif 'label' in train_df.columns:
                    label_col = 'label'
                else:
                    print(f"No label column found. Available columns: {train_df.columns.tolist()}")
                    return False
                
                print(f"Training dataset label distribution: {train_df[label_col].value_counts()}")
                
                # Standardize column names
                if 'Title' in train_df.columns:
                    train_df = train_df.rename(columns={'Title': 'title', 'Body': 'body', label_col: 'label'})
                
                # Filter HQ questions (handle both string and numeric labels)
                hq_mask = train_df['label'].isin(['HQ', 2])  # HQ can be 'HQ' or 2
                train_hq = train_df[hq_mask].copy()
                train_hq['source'] = 'train'
                
                if len(train_hq) > 0:
                    all_hq_questions.append(train_hq)
                    print(f"Loaded {len(train_hq)} HQ questions from training dataset")
                else:
                    print("No HQ questions found in training dataset")
            
            # Load testing dataset
            if os.path.exists(self.test_dataset_path):
                test_df = pd.read_csv(self.test_dataset_path)
                print(f"Testing dataset shape: {test_df.shape}")
                print(f"Testing dataset columns: {test_df.columns.tolist()}")
                
                # Handle different column naming conventions
                if 'Y' in test_df.columns:
                    label_col = 'Y'
                elif 'label' in test_df.columns:
                    label_col = 'label'
                else:
                    print(f"No label column found. Available columns: {test_df.columns.tolist()}")
                    return False
                
                print(f"Testing dataset label distribution: {test_df[label_col].value_counts()}")
                
                # Standardize column names
                if 'Title' in test_df.columns:
                    test_df = test_df.rename(columns={'Title': 'title', 'Body': 'body', label_col: 'label'})
                
                # Filter HQ questions (handle both string and numeric labels)
                hq_mask = test_df['label'].isin(['HQ', 2])  # HQ can be 'HQ' or 2
                test_hq = test_df[hq_mask].copy()
                test_hq['source'] = 'test'
                
                if len(test_hq) > 0:
                    all_hq_questions.append(test_hq)
                    print(f"Loaded {len(test_hq)} HQ questions from testing dataset")
                else:
                    print("No HQ questions found in testing dataset")
            
            if not all_hq_questions:
                print("No datasets found or no HQ questions available")
                print("Please check:")
                print("1. Dataset file paths are correct")
                print("2. Dataset files contain 'label' column")
                print("3. Labels are either 'HQ'/2 for high-quality questions")
                return False
            
            # Combine datasets
            combined_df = pd.concat(all_hq_questions, ignore_index=True)
            print(f"Combined dataset shape before deduplication: {combined_df.shape}")
            
            # Check required columns
            required_columns = ['title', 'body']
            missing_columns = [col for col in required_columns if col not in combined_df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                print(f"Available columns: {combined_df.columns.tolist()}")
                return False
            
            # Handle missing values
            combined_df['title'] = combined_df['title'].fillna('')
            combined_df['body'] = combined_df['body'].fillna('')
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['title', 'body'])
            print(f"Combined dataset shape after deduplication: {combined_df.shape}")
            
            combined_df['combined_text'] = combined_df['title'].astype(str) + " " + combined_df['body'].astype(str)
            combined_df['processed_text'] = combined_df['combined_text'].apply(preprocess_text)
            
            self.hq_questions = combined_df[['title', 'body', 'combined_text', 'processed_text', 'source']].to_dict('records')
            
            # Analyze HQ patterns
            print("Analyzing high-quality question patterns...")
            self.quality_analyzer.analyze_hq_patterns(self.hq_questions)
            
            print(f"Total HQ questions loaded: {len(self.hq_questions)}")
            return True
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
    
    def build_tfidf_index(self):
        """Build TF-IDF index"""
        if not self.hq_questions:
            return False
            
        texts = [q['processed_text'] for q in self.hq_questions]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print("TF-IDF index built successfully")
        return True
    
    def build_sbert_index(self):
        """Build SBERT embeddings"""
        if not self.hq_questions:
            return False
            
        texts = [q['processed_text'] for q in self.hq_questions]
        self.sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
        print("SBERT embeddings built successfully")
        return True

# Enhanced IR System with Pattern-Based Suggestions
class EnhancedIRSuggestionSystem:
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader
    
    def tfidf_similarity_search(self, query, top_k=3):
        """Find similar questions using TF-IDF"""
        if not self.dataset_loader.tfidf_vectorizer:
            return []
        
        processed_query = preprocess_text(query)
        query_vector = self.dataset_loader.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.dataset_loader.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'question': self.dataset_loader.hq_questions[idx],
                    'similarity': similarities[idx],
                    'method': 'TF-IDF'
                })
        return results
    
    def sbert_similarity_search(self, query, top_k=3):
        """Find similar questions using SBERT"""
        if self.dataset_loader.sbert_embeddings is None:
            return []
        
        processed_query = preprocess_text(query)
        query_embedding = self.dataset_loader.sbert_model.encode(processed_query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.dataset_loader.sbert_embeddings)[0]
        top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.3:
                results.append({
                    'question': self.dataset_loader.hq_questions[idx],
                    'similarity': score.item(),
                    'method': 'SBERT'
                })
        return results
    
    def hybrid_search(self, query, top_k=5):
        """Combine TF-IDF and SBERT results"""
        tfidf_results = self.tfidf_similarity_search(query, top_k)
        sbert_results = self.sbert_similarity_search(query, top_k)
        
        all_results = tfidf_results + sbert_results
        seen_questions = set()
        unique_results = []
        
        for result in all_results:
            question_text = result['question']['title']
            if question_text not in seen_questions:
                seen_questions.add(question_text)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        return unique_results[:top_k]

# Prediction Function
def predict_question_quality(title, body):
    combined_text = clean_html(title + " " + body)
    
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        label = label_encoder.inverse_transform([predicted_class])[0]
        return label, confidence

# Enhanced Suggestion Function
def suggest_improvements_with_pattern_analysis(title, body, ir_system):
    """Enhanced suggestion system with pattern-based analysis"""
    label, confidence = predict_question_quality(title, body)
    print(f"\nPredicted Question Quality: {label} (Confidence: {confidence:.2f})")
    
    if label == "HQ" and confidence < 0.80:
        print("Not confident enough to accept HQ. Suggesting improvements...")
        label = "LQ_EDIT"
    
    if label in ["LQ_EDIT", "LQ_CLOSE"]:
        print("\nAnalyzing your question against high-quality patterns...")
        
        # Pattern-based analysis
        quality_analysis = ir_system.dataset_loader.quality_analyzer.analyze_question_quality(title, body)

        
        # Always show pattern-based suggestions
        print(f"\nSpecific suggestions to improve your question:")
        suggestions = quality_analysis.get('suggestions', [])
        
        if suggestions:
            high_priority = [s for s in suggestions if s['priority'] == 'high']
            medium_priority = [s for s in suggestions if s['priority'] == 'medium']
            
            if high_priority:
                print("\nHIGH PRIORITY:")
                for i, suggestion in enumerate(high_priority, 1):
                    print(f"   {i}. {suggestion['message']}")
                    if 'example' in suggestion:
                        print(f"        Example: {suggestion['example']}")
            
            if medium_priority:
                print("\nMEDIUM PRIORITY:")
                for i, suggestion in enumerate(medium_priority, 1):
                    print(f"   {i}. {suggestion['message']}")
                    if 'example' in suggestion:
                        print(f"        Example: {suggestion['example']}")
        else:
            print("       Your question structure looks good, but could benefit from more specific details.")
        
        # General tips based on HQ patterns
        print(f"\nGeneral tips for high-quality questions:")
        patterns = ir_system.dataset_loader.quality_analyzer.hq_patterns
        if patterns:
            print(f"   • Average title length in HQ questions: {patterns['avg_title_length']:.1f} words")
            print(f"   • Average body length in HQ questions: {patterns['avg_body_length']:.1f} words")
            print(f"   • {patterns['has_code_examples']*100:.0f}% of HQ questions include code examples")
            print(f"   • {patterns['has_error_messages']*100:.0f}% of HQ questions include error messages")  # Fixed key
            # Show most common question types
            most_common_type = patterns['question_types'].most_common(1)[0] if patterns['question_types'] else None
            if most_common_type:
                print(f"   • Most common HQ question type: {most_common_type[0]} ({most_common_type[1]} questions)")
    
    else:
        print("Your question looks good. No suggestions needed.")

# Initialize Enhanced System
def initialize_enhanced_ir_system(train_dataset_path, test_dataset_path, force_rebuild=False):
    """Initialize the enhanced IR system"""
    dataset_loader = EnhancedHQDatasetLoader(train_dataset_path, test_dataset_path)
    
    print("Initializing enhanced question analysis system...")
    if dataset_loader.load_hq_questions():
        dataset_loader.build_tfidf_index()
        dataset_loader.build_sbert_index()
        return EnhancedIRSuggestionSystem(dataset_loader)
    else:
        print("Failed to load datasets")
        return None

# Example Usage
if __name__ == "__main__":
    # Initialize enhanced IR system
    ir_system = initialize_enhanced_ir_system(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    
    if ir_system:
        print("\n" + "="*60)
        print("ENHANCED QUESTION QUALITY ANALYZER")
        print("="*60)
        
        title = input("\nEnter the title of the question:\n> ")
        body = input("\nEnter the body of the question:\n> ")
        
        suggest_improvements_with_pattern_analysis(title, body, ir_system)
    else:
        print("Could not initialize system. Please check your dataset paths.")