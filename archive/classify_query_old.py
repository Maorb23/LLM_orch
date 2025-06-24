import json
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# This is a 
@dataclass
class QueryClassification:
    """Represents the classification result of a user query
    Example:
        QueryClassification(
            primary_category="CODE_TECHNICAL",
            confidence=0.85,
            secondary_categories=["MATHEMATICAL_SCIENTIFIC", "EDUCATIONAL_ACADEMIC"],
            complexity_level="INTERMEDIATE",
            response_format="CODE"
        )
    """
    primary_category: str
    confidence: float
    secondary_categories: List[str]
    complexity_level: str
    response_format: str


@dataclass
class ProcessedQuery:
    """Represents a query after preprocessing
    Example:        ProcessedQuery(
            original_query="Write a beautiful Python function to sort a list of numbers",
            normalized_query="write a python function to sort a list of numbers",
            key_terms=["python", "function", "sort", "list", "numbers"],
            action_verbs=["write", "sort"],
    """
    original_query: str
    normalized_query: str
    key_terms: List[str]
    action_verbs: List[str]
    features: Dict


class QueryPreprocessor:
    """Handles text normalization and feature extraction from user queries"""
    
    def __init__(self):
        # Terms that should be expanded/normalized
        self.shortened_tech_terms = {
            # Programming Languages
            'js': 'javascript',
            'py': 'python',
            'ts': 'typescript',
            'cpp': 'c++',
            'cs': 'c#',
            'rb': 'ruby',
            'php': 'php',
            'go': 'golang',
            'rs': 'rust',
            'kt': 'kotlin',
            'swift': 'swift',
            
            # Frameworks & Libraries
            'react': 'react',
            'vue': 'vue',
            'angular': 'angular',
            'django': 'django',
            'flask': 'flask',
            'express': 'express',
            'spring': 'spring',
            'laravel': 'laravel',
            'rails': 'ruby on rails',
            'tf': 'tensorflow',
            'pytorch': 'pytorch',
            'sklearn': 'scikit-learn',
            'np': 'numpy',
            'pd': 'pandas',
            
            # Technologies & Concepts
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'nn': 'neural network',
            'cnn': 'convolutional neural network',
            'rnn': 'recurrent neural network',
            'lstm': 'long short-term memory',
            'gan': 'generative adversarial network',
            'llm': 'large language model',
            'gpt': 'generative pre-trained transformer',
            'bert': 'bidirectional encoder representations from transformers',
            
            # Data & Databases
            'db': 'database',
            'sql': 'structured query language',
            'nosql': 'nosql',
            'mysql': 'mysql',
            'postgres': 'postgresql',
            'mongo': 'mongodb',
            'redis': 'redis',
            'elasticsearch': 'elasticsearch',
            
            # Web Technologies
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'xml': 'xml',
            'api': 'application programming interface',
            'rest': 'representational state transfer',
            'graphql': 'graphql',
            'jwt': 'json web token',
            'oauth': 'oauth',
            'cors': 'cross-origin resource sharing',
            
            # DevOps & Tools
            'git': 'git',
            'docker': 'docker',
            'k8s': 'kubernetes',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'azure': 'microsoft azure',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'ide': 'integrated development environment',
            'cli': 'command line interface',
            'gui': 'graphical user interface',
            
            # Operating Systems
            'os': 'operating system',
            'linux': 'linux',
            'ubuntu': 'ubuntu',
            'centos': 'centos',
            'macos': 'macos',
            'windows': 'windows',
            
            # Algorithms & Data Structures
            'algo': 'algorithm',
            'ds': 'data structure',
            'bfs': 'breadth first search',
            'dfs': 'depth first search',
            'dp': 'dynamic programming',
        }
        
        # Terms that should NOT be normalized (preserve original form)
        # These are important technical terms that should remain as-is
        self.preserve_tech_terms = {
            # Programming Languages (keep original case/form)
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'ruby', 
            'php', 'golang', 'rust', 'kotlin', 'swift', 'scala', 'perl',
            
            # Frameworks & Libraries
            'react', 'vue', 'angular', 'django', 'flask', 'express', 'spring',
            'laravel', 'tensorflow', 'pytorch', 'scikit-learn', 'numpy', 'pandas',
            'matplotlib', 'seaborn', 'opencv', 'keras', 'fastapi', 'streamlit',
            
            # Technologies
            'artificial intelligence', 'machine learning', 'deep learning',
            'natural language processing', 'computer vision', 'neural network',
            'convolutional neural network', 'recurrent neural network',
            'long short-term memory', 'generative adversarial network',
            'large language model', 'transformer', 'attention mechanism',
            
            # Databases
            'database', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite',
            'oracle', 'cassandra', 'elasticsearch', 'solr',
            
            # Web Technologies
            'html', 'css', 'json', 'xml', 'yaml', 'api', 'rest', 'graphql',
            'websocket', 'http', 'https', 'tcp', 'udp', 'ssl', 'tls',
            
            # Cloud & DevOps
            'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
            'amazon web services', 'google cloud platform', 'microsoft azure',
            'continuous integration', 'continuous deployment',
            
            # Algorithms & Concepts
            'algorithm', 'data structure', 'recursion', 'sorting', 'searching',
            'hashing', 'encryption', 'compression', 'optimization',
            'breadth first search', 'depth first search', 'dynamic programming',
            
            # Security
            'authentication', 'authorization', 'encryption', 'decryption',
            'hashing', 'digital signature', 'certificate', 'firewall',
            
            # Software Engineering
            'object oriented programming', 'functional programming',
            'design pattern', 'microservices', 'monolith', 'architecture',
            'refactoring', 'debugging', 'testing', 'unit test', 'integration test'
        }
        
        # Create comprehensive technical terms list for feature extraction
        self.all_tech_terms = self._build_comprehensive_tech_terms()
        
        # Common action verbs that indicate query intent
        self.action_verbs = [
            'write', 'create', 'build', 'generate', 'explain', 'analyze',
            'debug', 'fix', 'optimize', 'compare', 'summarize', 'teach',
            'implement', 'design', 'develop', 'code', 'program', 'solve',
            'help', 'show', 'demonstrate', 'convert', 'transform', 'migrate'
        ]
    
    def _build_comprehensive_tech_terms(self) -> set:
        """
        Build a comprehensive set of all technical terms for feature extraction
        Combines normalized terms + preserved terms + original shortened forms
        """
        all_terms = set()
        
        # Add all normalized/expanded terms
        all_terms.update(self.shortened_tech_terms.values())
        
        # Add all preserved terms
        all_terms.update(self.preserve_tech_terms)
        
        # Add original shortened forms (they might appear in queries)
        all_terms.update(self.shortened_tech_terms.keys())
        
        # Add common variations and plurals
        additional_terms = set()
        for term in all_terms.copy():
            if not term.endswith('s'):
                additional_terms.add(term + 's')  # plurals
            if ' ' in term:
                # Add acronym versions of multi-word terms
                words = term.split()
                if len(words) <= 4:  # Only for reasonable length
                    acronym = ''.join(word[0].lower() for word in words)
                    additional_terms.add(acronym)
        
        all_terms.update(additional_terms)
        
        return all_terms
    
    def normalize_text(self, query: str) -> str:
        """
        Normalize the input query by expanding abbreviations while preserving important terms
        
        Args:
            query: Raw user input
            
        Returns:
            Normalized query string
        """
        import re
        
        # Basic cleaning
        normalized = query.strip().lower()
        
        # Expand common abbreviations (but preserve context)
        words = normalized.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation for matching but preserve it
            clean_word = re.sub(r'[^\w]', '', word)
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
            # Check if word should be expanded
            if clean_word in self.shortened_tech_terms:
                expanded = self.shortened_tech_terms[clean_word]
                normalized_words.append(expanded + punctuation)
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def extract_features(self, query: str) -> Dict:
        """
        Extract key features from the query for classification using comprehensive tech terms
        
        Args:
            query: Normalized query string
            
        Returns:
            Dictionary containing extracted features
        """
        import re
        
        words = query.lower().split()
        
        # Find technical keywords using comprehensive list
        tech_keywords = []
        for word in words:
            # Clean word for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.all_tech_terms:
                tech_keywords.append(clean_word)
        
        # Also check for multi-word technical terms
        query_lower = query.lower()
        for term in self.all_tech_terms:
            if ' ' in term and term in query_lower:
                tech_keywords.append(term)
        
        # Remove duplicates while preserving order
        tech_keywords = list(dict.fromkeys(tech_keywords))
        
        # Find action verbs
        action_verbs = [word for word in words if word in self.action_verbs]
        
        # Basic complexity indicators
        complexity_indicators = {
            'has_code_markers': bool(re.search(r'[{}()\[\];]', query)),
            'has_special_chars': bool(re.search(r'[<>@#$%^&*]', query)),
            'word_count': len(words),
            'char_count': len(query),
            'has_numbers': bool(re.search(r'\d', query)),
            'has_urls': bool(re.search(r'http[s]?://|www\.', query)),
            'question_words': len([w for w in words if w in ['what', 'how', 'why', 'when', 'where', 'which']]),
            'tech_density': len(tech_keywords) / len(words) if words else 0
        }
        
        return {
            'tech_keywords': tech_keywords,
            'action_verbs': action_verbs,
            'word_count': len(words),
            'complexity': complexity_indicators,
            'has_tech_terms': len(tech_keywords) > 0,
            'tech_term_count': len(tech_keywords),
            'primary_tech_category': self._categorize_tech_terms(tech_keywords)
        }
    
    def _categorize_tech_terms(self, tech_keywords: List[str]) -> str:
        """Categorize the primary technology focus based on found keywords"""
        
        categories = {
            'programming': ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'php', 'golang', 'rust'],
            'web_development': ['html', 'css', 'react', 'vue', 'angular', 'api', 'rest', 'graphql'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'numpy', 'pandas'],
            'database': ['database', 'mysql', 'postgresql', 'mongodb', 'redis', 'sql'],
            'devops': ['docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'git'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin']
        }
        
        category_scores = {}
        for category, terms in categories.items():
            score = sum(1 for keyword in tech_keywords if keyword in terms or any(keyword in term for term in terms))
            if score > 0:
                category_scores[category] = score
        
        return max(category_scores, key=category_scores.get) if category_scores else 'general'

    # In QueryPreprocessor class, ADD this method:

    def process(self, query: str) -> ProcessedQuery:
        """
        Main preprocessing pipeline with logging
        
        Args:
            query: Raw user input string
            
        Returns:
            ProcessedQuery object with normalized text and extracted features
        """
        try:
            # Step 1: Normalize text
            normalized = self.normalize_text(query)
            
            # Step 2: Extract features
            features = self.extract_features(normalized)
            
            # Step 3: Extract key terms and action verbs from features
            key_terms = features.get('tech_keywords', [])
            action_verbs = features.get('action_verbs', [])
            
            return ProcessedQuery(
                original_query=query,
                normalized_query=normalized,
                key_terms=key_terms,
                action_verbs=action_verbs,
                features=features
            )
            
        except Exception as e:
            # Fallback processing if anything fails
            return ProcessedQuery(
                original_query=query,
                normalized_query=query.lower().strip(),
                key_terms=[],
                action_verbs=[],
                features={"word_count": len(query.split()), "error": str(e)}
            )

class QueryClassifier:
    """
    Classifies queries using embedding similarity and rule-based fallbacks
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
        
        # Define category patterns and keywords
        self.category_patterns = {
            'CODE_TECHNICAL': [
                'python', 'javascript', 'java', 'code', 'programming', 'debug',
                'api', 'database', 'algorithm', 'function', 'class', 'variable'
            ],
            'MATHEMATICAL_SCIENTIFIC': [
                'calculate', 'equation', 'formula', 'statistics', 'physics',
                'chemistry', 'mathematics', 'solve', 'analyze data',
                # ABSORBED: Research & Analysis (analytical/scientific)
                'research', 'analyze', 'compare', 'study', 'trend', 'synthesis',
                'investigation', 'correlation', 'hypothesis', 'methodology'
            ],
            'CREATIVE_ARTISTIC': [
                'write story', 'poem', 'creative', 'art', 'music', 'design',
                'narrative', 'character', 'plot', 'rhyme'
            ],
            'BUSINESS_PROFESSIONAL': [
                'business', 'marketing', 'strategy', 'professional', 'email',
                'presentation', 'meeting', 'proposal', 'budget',
                # ABSORBED: Legal & Compliance (professional domain)
                'legal', 'contract', 'compliance', 'regulation', 'policy',
                'gdpr', 'privacy', 'terms', 'agreement', 'liability'
            ],
            'EDUCATIONAL_ACADEMIC': [
                'explain', 'teach', 'learn', 'study', 'academic', 'research',
                'thesis', 'assignment', 'homework', 'course',
                # ABSORBED: Health & Wellness (educational/informational)
                'health', 'wellness', 'fitness', 'nutrition', 'diet',
                'exercise', 'mental health', 'therapy', 'medical', 'symptoms'
            ],
            'CONVERSATIONAL_ADVICE': [
                'advice', 'help me', 'what should i', 'how do i', 'personal',
                'relationship', 'lifestyle', 'recommend'
            ]
        }
        
        # Add specialized sub-category detection For future enhancements
        # self.specialized_subcategories = {
        #     'research_analysis': [
        #         'compare', 'analyze', 'trend', 'market', 'research', 'study',
        #         'synthesis', 'investigation', 'data analysis', 'survey'
        #     ],
        #     'legal_compliance': [
        #         'legal', 'law', 'contract', 'gdpr', 'compliance', 'regulation',
        #         'policy', 'terms', 'agreement', 'privacy', 'copyright'
        #     ],
        #     'health_wellness': [
        #         'health', 'medical', 'fitness', 'nutrition', 'diet', 'exercise',
        #         'wellness', 'mental health', 'therapy', 'symptoms', 'treatment'
        #     ]
        # }
        
        # Pre-compute category embeddings if model available
        self.category_embeddings = {}
        if self.embedding_model:
            self._compute_category_embeddings()
    
    def _compute_category_embeddings(self):
        """Pre-compute embeddings for each category using representative text"""
        category_descriptions = {
            'CODE_TECHNICAL': "programming coding software development debugging algorithms",
            'MATHEMATICAL_SCIENTIFIC': "mathematics science calculations equations data analysis",
            'CREATIVE_ARTISTIC': "creative writing stories poems art music design",
            'BUSINESS_PROFESSIONAL': "business strategy marketing professional communication",
            'EDUCATIONAL_ACADEMIC': "education teaching learning academic research explanation",
            'CONVERSATIONAL_ADVICE': "personal advice help guidance recommendations"
        }
        
        try:
            for category, description in category_descriptions.items():
                embedding = self.embedding_model.encode(description)
                self.category_embeddings[category] = embedding
            logger.info("Category embeddings computed successfully.")

        except Exception as e:
            print(f"Warning: Could not compute category embeddings: {e}")
    
    def _classify_by_rules(self, processed_query: ProcessedQuery) -> Tuple[str, float]:
        """
        Rule-based classification using keyword matching
        
        Args:
            processed_query: Preprocessed query object. Fetched by QueryPreprocessor
            
        Returns:
            Tuple of (category, confidence_score)
        """
        query_text = processed_query.normalized_query
        scores = {}
        
        # Calculate scores for each category based on keyword matches
        for category, keywords in self.category_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in query_text:
                    score += 1
            
            # Normalize by number of keywords in category
            scores[category] = score / len(keywords) if keywords else 0
        
        if not scores or max(scores.values()) == 0:
            return 'CONVERSATIONAL_ADVICE', 0.3  # Default fallback
        
        best_category = max(scores, key=scores.get)
        confidence = min(scores[best_category] * 2, 1.0)  # Scale confidence
        logging.info(f"Rule-based classification: {best_category} with confidence {confidence:.2f}")
        
        return best_category, confidence
    
    def _classify_by_embedding(self, processed_query: ProcessedQuery) -> Tuple[str, float]:
        """
        Embedding-based classification using cosine similarity
        
        Args:
            processed_query: Preprocessed query object
            
        Returns:
            Tuple of (category, confidence_score)
        """
        if not self.embedding_model or not self.category_embeddings:
            logger.warning("Embedding model or category embeddings not available, falling back to rule-based classification.")
            return self._classify_by_rules(processed_query)
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode(processed_query.normalized_query)
            logger.info(f"Query embedding computed: {query_embedding[:5]}...")  # Log first 5 values
            # Calculate similarities with each category
            similarities = {}
            for category, cat_embedding in self.category_embeddings.items():
                similarity = np.dot(query_embedding, cat_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cat_embedding)
                )
                similarities[category] = similarity
            
            best_category = max(similarities, key=similarities.get)
            logger.info(f"Embedding classification: {best_category} with similarity {similarities[best_category]:.2f}")
            confidence = float(similarities[best_category])
            logger.info(f"Embedding confidence score: {confidence:.2f}")    
            
            return best_category, confidence
            
        except Exception as e:
            print(f"Warning: Embedding classification failed: {e}")
            return self._classify_by_rules(processed_query)
    
    def _determine_complexity(self, processed_query: ProcessedQuery) -> str:
        """
        Determine query complexity level based on features
        
        Args:
            processed_query: Preprocessed query object
            
        Returns:
            Complexity level string
        """
        features = processed_query.features
        word_count = features.get('word_count', 0)
        
        # Simple heuristic for complexity
        if word_count < 5:
            return 'BASIC'
        elif word_count < 20:
            if features.get('complexity', {}).get('has_code_markers') or \
               features.get('complexity', {}).get('has_math_symbols'):
                return 'INTERMEDIATE'
            return 'BASIC'
        else:
            return 'ADVANCED'
    
    def _determine_response_format(self, processed_query: ProcessedQuery, category: str) -> str:
        """
        Determine expected response format based on query and category
        
        Args:
            processed_query: Preprocessed query object
            category: Classified category
            
        Returns:
            Response format string
        """
        features = processed_query.features
        
        # Check for explicit format requests
        if features.get('complexity', {}).get('has_code_markers'):
            return 'CODE'
        
        # Category-based format prediction
        format_mapping = {
            'CODE_TECHNICAL': 'CODE',
            'MATHEMATICAL_SCIENTIFIC': 'STRUCTURED',
            'CREATIVE_ARTISTIC': 'NARRATIVE',
            'BUSINESS_PROFESSIONAL': 'STRUCTURED',
            'EDUCATIONAL_ACADEMIC': 'NARRATIVE',
            'CONVERSATIONAL_ADVICE': 'INTERACTIVE'
        }
        
        return format_mapping.get(category, 'NARRATIVE')
    
    def classify(self, processed_query: ProcessedQuery) -> QueryClassification:
        """
        Main classification method combining multiple approaches
        
        Args:
            processed_query: Preprocessed query object
            
        Returns:
            QueryClassification object with results
        """
        # Try embedding-based classification first
        primary_category, confidence = self._classify_by_embedding(processed_query)
        
        # If confidence is low, also try rule-based as backup
        if confidence < 0.6:
            rule_category, rule_confidence = self._classify_by_rules(processed_query)
            if rule_confidence > confidence:
                primary_category, confidence = rule_category, rule_confidence
        
        # Determine other attributes
        complexity = self._determine_complexity(processed_query)
        response_format = self._determine_response_format(processed_query, primary_category)
        
        # For now, secondary categories are empty (can be enhanced later)
        secondary_categories = []
        
        return QueryClassification(
            primary_category=primary_category,
            confidence=confidence,
            secondary_categories=secondary_categories,
            complexity_level=complexity,
            response_format=response_format
        )


class LLMRouter:
    """
    Routes queries to appropriate LLMs based on classification results
    """
    
    def __init__(self, api_key_path: str):
        """
        Args:
            api_key_path: Path to file containing Nebius API key
        """
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
            
            self.client = OpenAI(
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
        
        # Define specialized models for different categories
        self.specialized_models = {
            'CODE_TECHNICAL': 'Qwen/Qwen3-235B-A22B',  # Use best model for coding
            'MATHEMATICAL_SCIENTIFIC': 'deepseek-ai/DeepSeek-V3',
            'CREATIVE_ARTISTIC': 'google/gemma-2-2b-it',
            'BUSINESS_PROFESSIONAL': 'google/gemma-2-2b-it',
            'EDUCATIONAL_ACADEMIC': 'mistralai/Mistral-Nemo-Instruct-2407',
            'CONVERSATIONAL_ADVICE': 'google/gemma-2-2b-it'
        }
        
        self.default_model = 'Qwen/Qwen3-235B-A22B'
        self.confidence_threshold = 0.6
    
    def _select_model(self, classification: QueryClassification) -> str:
        """
        Select appropriate model based on classification
        
        Args:
            classification: Query classification results
            
        Returns:
            Model name to use
        """
        if classification.confidence >= self.confidence_threshold:
            return self.specialized_models.get(
                classification.primary_category, 
                self.default_model
            )
        return self.default_model
    
    def _build_system_prompt(self, classification: QueryClassification) -> str:
        """
        Build system prompt based on classification results
        
        Args:
            classification: Query classification results
            
        Returns:
            System prompt string
        """
        base_prompt = "You are a helpful AI assistant."
        
        # Add category-specific instructions
        category_prompts = {
            'CODE_TECHNICAL': "You specialize in programming and technical problem-solving. Provide clear, working code examples with explanations.",
            'MATHEMATICAL_SCIENTIFIC': "You excel at mathematical and scientific reasoning. Show your work step-by-step.",
            'CREATIVE_ARTISTIC': "You are creative and imaginative. Focus on storytelling and artistic expression.",
            'BUSINESS_PROFESSIONAL': "You provide professional business advice with practical, actionable insights.",
            'EDUCATIONAL_ACADEMIC': "You are an excellent teacher. Explain concepts clearly with examples.",
            'CONVERSATIONAL_ADVICE': "You are empathetic and helpful with personal matters. Provide thoughtful advice."
        }
        
        category_instruction = category_prompts.get(classification.primary_category, "")
        if category_instruction:
            base_prompt += f" {category_instruction}"
        
        # Add complexity-based instructions
        if classification.complexity_level == 'BASIC':
            base_prompt += " Keep your explanation simple and beginner-friendly."
        elif classification.complexity_level == 'ADVANCED':
            base_prompt += " Provide detailed, comprehensive analysis suitable for experts."
        
        return base_prompt
    
    def generate_response(self, query: str, classification: QueryClassification) -> str:
        """
        Generate response using selected LLM
        
        Args:
            query: Original user query
            classification: Classification results
            
        Returns:
            Generated response string
        """
        model = self._select_model(classification)
        system_prompt = self._build_system_prompt(classification)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Sorry, I encountered an error generating a response: {str(e)}"


class QueryJudge:
    """
    Judges the quality of LLM responses using another LLM as evaluator
    """
    
    def __init__(self, api_key_path: str, model_name: str = "Qwen/Qwen3-235B-A22B"):
        """
        Args:
            api_key_path: Path to file containing Nebius API key
            model_name: Model to use for judging
        """
        self.model_name = model_name
        
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
            
            self.client = OpenAI(
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize judge client: {e}")
    
    def judge_response(self, query: str, response: str) -> Dict:
        """
        Judge the quality of an LLM response
        
        Args:
            query: Original user query
            response: LLM generated response
            
        Returns:
            Dictionary with score, justification, and correctness
        """
        judge_prompt = f"""
        Please evaluate the following LLM response to a user query on a scale of 1-5:
        
        User Query: {query}
        LLM Response: {response}
        
        Consider:
        - Relevance to the query
        - Accuracy of information
        - Clarity and helpfulness
        - Completeness of the answer
        
        Respond with a JSON object containing:
        - "score": integer from 1-5
        - "justification": explanation of your rating
        - "is_correct": boolean indicating if the response is factually correct
        """
        
        try:
            judge_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            
            # Try to parse JSON response
            response_text = judge_response.choices[0].message.content
            try:
                judgment = json.loads(response_text)
                return judgment
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "score": 3,
                    "justification": "Could not parse judge response",
                    "is_correct": True
                }
                
        except Exception as e:
            return {
                "score": 3,
                "justification": f"Error in judging: {str(e)}",
                "is_correct": True
            }


class LLMQueryRouter:
    """
    Main class that orchestrates the entire query routing pipeline
    """
    
    def __init__(self, api_key_path: str):
        """
        Initialize all components of the routing system
        
        Args:
            api_key_path: Path to file containing Nebius API key
        """
        self.preprocessor = QueryPreprocessor()
        self.classifier = QueryClassifier()
        self.router = LLMRouter(api_key_path)
        self.judge = QueryJudge(api_key_path)
    
    def process_query(self, query: str, include_judgment: bool = False) -> Dict:
        """
        Main pipeline: preprocess -> classify -> route -> generate -> (judge)
        
        Args:
            query: Raw user query
            include_judgment: Whether to include quality judgment
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            processed_query = self.preprocessor.process(query)
            
            # Step 2: Classify query
            classification = self.classifier.classify(processed_query)
            
            # Step 3: Generate response
            response = self.router.generate_response(query, classification)
            
            # Step 4: Optional judgment
            judgment = None
            if include_judgment:
                judgment = self.judge.judge_response(query, response)
            
            processing_time = time.time() - start_time
            
            return {
                'response': response,
                'classification': {
                    'category': classification.primary_category,
                    'confidence': classification.confidence,
                    'complexity': classification.complexity_level,
                    'format': classification.response_format
                },
                'judgment': judgment,
                'processing_time_seconds': processing_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'response': f"I'm sorry, I encountered an error processing your query: {str(e)}",
                'error': str(e),
                'processing_time_seconds': time.time() - start_time,
                'success': False
            }


# Example usage
if __name__ == "__main__":
    # Initialize the router
    router = LLMQueryRouter("Nebius_api_key.txt")
    
    # Process a sample query
    test_query = "What are the benefits of using renewable energy sources in modern economies?"
    result = router.process_query(test_query, include_judgment=True)
    
    print(f"Query: {test_query}")
    print(f"Category: {result['classification']['category']}")
    print(f"Confidence: {result['classification']['confidence']:.2f}")
    print(f"Response: {result['response']}")
    
    if result.get('judgment'):
        print(f"Quality Score: {result['judgment']['score']}/5")
        print(f"Justification: {result['judgment']['justification']}")