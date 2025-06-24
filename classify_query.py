# filepath: c:\Users\maorb\work\Tryaii\classify_query.py
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
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
        logger.info("Initializing QueryPreprocessor")
        
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
        logger.info(f"Built comprehensive tech terms list with {len(self.all_tech_terms)} terms")
        
        # Common action verbs that indicate query intent
        self.action_verbs = [
            'write', 'create', 'build', 'generate', 'explain', 'analyze',
            'debug', 'fix', 'optimize', 'compare', 'summarize', 'teach',
            'implement', 'design', 'develop', 'code', 'program', 'solve',
            'help', 'show', 'demonstrate', 'convert', 'transform', 'migrate'
        ]
        
        logger.info("QueryPreprocessor initialization completed")
    
    def _build_comprehensive_tech_terms(self) -> set:
        """
        Build a comprehensive set of all technical terms for feature extraction
        Combines normalized terms + preserved terms + original shortened forms
        """
        logger.info("Building comprehensive technical terms list")
        
        all_terms = set()
        
        # Add all normalized/expanded terms
        all_terms.update(self.shortened_tech_terms.values())
        logger.info(f"Added {len(self.shortened_tech_terms.values())} normalized terms")
        
        # Add all preserved terms
        all_terms.update(self.preserve_tech_terms)
        logger.info(f"Added {len(self.preserve_tech_terms)} preserved terms")
        
        # Add original shortened forms (they might appear in queries)
        all_terms.update(self.shortened_tech_terms.keys())
        logger.info(f"Added {len(self.shortened_tech_terms.keys())} shortened forms")
        
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
        logger.info(f"Added {len(additional_terms)} variations and plurals")
        logger.info(f"Total comprehensive tech terms: {len(all_terms)}")
        
        return all_terms
    
    def normalize_text(self, query: str) -> str:
        """
        Normalize the input query by expanding abbreviations while preserving important terms
        
        Args:
            query: Raw user input
            
        Returns:
            Normalized query string
        """
        logger.info(f"Starting text normalization for query: '{query[:50]}...'")
        import re
        
        # Basic cleaning
        normalized = query.strip().lower()
        logger.info(f"After basic cleaning: '{normalized[:50]}...'")
        
        # Expand common abbreviations (but preserve context)
        words = normalized.split()
        normalized_words = []
        expansions_made = 0
        
        for word in words:
            # Remove punctuation for matching but preserve it
            clean_word = re.sub(r'[^\w]', '', word)
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
            # Check if word should be expanded
            if clean_word in self.shortened_tech_terms:
                expanded = self.shortened_tech_terms[clean_word]
                normalized_words.append(expanded + punctuation)
                logger.info(f"Expanded '{clean_word}' to '{expanded}'")
                expansions_made += 1
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        logger.info(f"Text normalization completed. Made {expansions_made} expansions")
        logger.info(f"Final normalized text: '{normalized[:100]}...'")
        
        return normalized
    
    def extract_features(self, query: str) -> Dict:
        """
        Extract key features from the query for classification using comprehensive tech terms
        
        Args:
            query: Normalized query string
            
        Returns:
            Dictionary containing extracted features
        """
        logger.info(f"Starting feature extraction for query: '{query[:50]}...'")
        import re
        
        words = query.lower().split()
        logger.info(f"Query split into {len(words)} words")
        
        # Find technical keywords using comprehensive list
        tech_keywords = []
        for word in words:
            # Clean word for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.all_tech_terms:
                tech_keywords.append(clean_word)
                logger.info(f"Found tech keyword: '{clean_word}'")
        
        # Also check for multi-word technical terms
        query_lower = query.lower()
        multiword_matches = 0
        for term in self.all_tech_terms:
            if ' ' in term and term in query_lower:
                tech_keywords.append(term)
                multiword_matches += 1
                logger.info(f"Found multi-word tech term: '{term}'")
        
        logger.info(f"Found {multiword_matches} multi-word technical terms")
        
        # Remove duplicates while preserving order
        original_count = len(tech_keywords)
        tech_keywords = list(dict.fromkeys(tech_keywords))
        logger.info(f"Removed {original_count - len(tech_keywords)} duplicate tech keywords")
        
        # Find action verbs
        action_verbs = [word for word in words if word in self.action_verbs]
        logger.info(f"Found action verbs: {action_verbs}")
        
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
        
        logger.info(f"Complexity indicators: {complexity_indicators}")
        
        primary_tech_category = self._categorize_tech_terms(tech_keywords)
        logger.info(f"Primary tech category determined: {primary_tech_category}")
        
        features = {
            'tech_keywords': tech_keywords,
            'action_verbs': action_verbs,
            'word_count': len(words),
            'complexity': complexity_indicators,
            'has_tech_terms': len(tech_keywords) > 0,
            'tech_term_count': len(tech_keywords),
            'primary_tech_category': primary_tech_category
        }
        
        logger.info(f"Feature extraction completed. Found {len(tech_keywords)} tech terms, tech density: {complexity_indicators['tech_density']:.2f}")
        
        return features
    
    def _categorize_tech_terms(self, tech_keywords: List[str]) -> str:
        """Categorize the primary technology focus based on found keywords"""
        logger.info(f"Categorizing tech terms: {tech_keywords}")
        
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
                logger.info(f"Category '{category}' scored {score} points")
        
        result = max(category_scores, key=category_scores.get) if category_scores else 'general'
        logger.info(f"Primary tech category determined: {result} (scores: {category_scores})")
        
        return result

    # In QueryPreprocessor class, ADD this method:

    def process(self, query: str) -> ProcessedQuery:
        """
        Main preprocessing pipeline with logging
        
        Args:
            query: Raw user input string
            
        Returns:
            ProcessedQuery object with normalized text and extracted features
        """
        logger.info(f"Starting query preprocessing pipeline for: '{query[:100]}...'")
        
        try:
            # Step 1: Normalize text
            logger.info("Step 1: Text normalization")
            normalized = self.normalize_text(query)
            
            # Step 2: Extract features
            logger.info("Step 2: Feature extraction")
            features = self.extract_features(normalized)
            
            # Step 3: Extract key terms and action verbs from features
            logger.info("Step 3: Extract key terms and action verbs")
            key_terms = features.get('tech_keywords', [])
            action_verbs = features.get('action_verbs', [])
            
            processed_query = ProcessedQuery(
                original_query=query,
                normalized_query=normalized,
                key_terms=key_terms,
                action_verbs=action_verbs,
                features=features
            )
            
            logger.info(f"Query preprocessing completed successfully")
            logger.info(f"Key terms found: {key_terms}")
            logger.info(f"Action verbs found: {action_verbs}")
            logger.info(f"Word count: {features.get('word_count', 0)}")
            
            return processed_query
            
        except Exception as e:
            logger.warning(f"Error during query preprocessing: {str(e)}")
            logger.warning("Falling back to basic processing")
            
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
        logger.info(f"Initializing QueryClassifier with model: {model_name}")
        
        # Load sentence transformer for embeddings
        try:
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            logger.warning("Falling back to rule-based classification only")
            self.embedding_model = None
        
        # Define category patterns and keywords
        self.category_patterns = {
            'CODE_TECHNICAL': [
                'python', 'javascript', 'java', 'code', 'programming', 'debug',
                'api', 'database', 'algorithm', 'function', 'class', 'variable',
                'bug', 'error', 'compile', 'syntax'
            ],
            'MATHEMATICAL_SCIENTIFIC': [
                'calculate', 'equation', 'formula', 'statistics', 'physics',
                'chemistry', 'mathematics', 'solve', 'analyze data',
                # ABSORBED: Research & Analysis (analytical/scientific)
                'research', 'analyze', 'compare', 'study', 'trend', 'synthesis',
                'investigation', 'correlation', 'hypothesis', 'methodology',
                'renewable energy', 'climate', 'environment', 'sustainability',
                'economic impact', 'benefits', 'advantages', 'effects'
            ],
            'CREATIVE_ARTISTIC': [
                'write story', 'poem', 'creative', 'art', 'music', 'design',
                'narrative', 'character', 'plot', 'rhyme', 'creative writing'
            ],
            'BUSINESS_PROFESSIONAL': [
                'business', 'marketing', 'strategy', 'professional', 'email',
                'presentation', 'meeting', 'proposal', 'budget', 'revenue',
                # ABSORBED: Legal & Compliance (professional domain)
                'legal', 'contract', 'compliance', 'regulation', 'policy',
                'gdpr', 'privacy', 'terms', 'agreement', 'liability', 'lawsuit',
                'intellectual property', 'trademark', 'copyright', 'patent',
            ],
            'EDUCATIONAL_ACADEMIC': [
                'explain', 'teach', 'learn', 'study', 'academic', 'research',
                'thesis', 'assignment', 'homework', 'course', 'benefits of',
                'advantages of', 'what are', 'how does', 'why is',
                # ABSORBED: Health & Wellness (educational/informational)
                'health', 'wellness', 'fitness', 'nutrition', 'diet',
                'exercise', 'mental health', 'therapy', 'medical', 'symptoms'
            ],
            'CONVERSATIONAL_ADVICE': [
                'advice', 'help me', 'what should i', 'how do i', 'personal',
                'relationship', 'lifestyle', 'recommend', 'opinion', 'suggest'
            ]
        }
        
        logger.info(f"Initialized category patterns for {len(self.category_patterns)} categories")
        for category, keywords in self.category_patterns.items():
            logger.info(f"Category '{category}': {len(keywords)} keywords")
        
        # Pre-compute category embeddings if model available
        self.category_embeddings = {}
        if self.embedding_model:
            logger.info("Computing category embeddings...")
            self._compute_category_embeddings()
            logger.info(f"Category embeddings computed for {len(self.category_embeddings)} categories")
        else:
            logger.warning("Skipping category embeddings - no embedding model available")

        self.llm_client = None
        api_key_path = "Nebius_api_key.txt"  # Path to your API key file
        if api_key_path:
            try:
                logger.info(f"Initializing LLM client for classification from: {api_key_path}")
                with open(api_key_path, "r") as f:
                    api_key = f.read().strip()
                
                self.llm_client = OpenAI(
                    base_url="https://api.studio.nebius.ai/v1/",
                    api_key=api_key,
                )
                logger.info("LLM client for classification initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
                logger.warning("Will use embedding and rule-based classification only")
        else:
            logger.info("No API key provided - LLM classification disabled")
        
        # Classification model for LLM
        self.classification_model = "Qwen/Qwen3-235B-A22B"  # Use the best model for classification
    

        logger.info("QueryClassifier initialization completed")

    
    def _compute_category_embeddings(self):
        """Pre-compute embeddings using multiple representative examples per category"""
        logger.info("Computing enhanced category embeddings")
        
        # Multiple diverse examples per category for better representation
        category_examples = {
            'CODE_TECHNICAL': [
                "write a python function to sort a list",
                "debug this javascript error in my code",
                "how to implement a binary search algorithm",
                "optimize SQL query performance",
                "fix compilation error in C++ program",
                "create REST API endpoint",
                "database schema design patterns"
            ],
            'MATHEMATICAL_SCIENTIFIC': [
                "calculate the derivative of this function",
                "analyze statistical correlation in dataset",
                "solve differential equation using numerical methods",
                "what are the benefits of renewable energy in economics",
                "research methodology for scientific experiments",
                "data analysis and visualization techniques",
                "mathematical modeling of population growth"
            ],
            'EDUCATIONAL_ACADEMIC': [
                "explain quantum physics concepts simply",
                "teach me about photosynthesis process",
                "what are the causes of world war 2",
                "help me understand machine learning basics",
                "summarize the main points of this research paper",
                "create study guide for biology exam",
                "explain the economic benefits of renewable energy"
            ],
            'CREATIVE_ARTISTIC': [
                "write a short story about space travel",
                "create poem about autumn leaves",
                "design logo for coffee shop",
                "compose lyrics for a love song",
                "brainstorm creative marketing campaign ideas",
                "write character development for novel"
            ],
            'BUSINESS_PROFESSIONAL': [
                "draft professional email to client",
                "create business proposal for new project",
                "analyze market trends and competition",
                "develop marketing strategy for product launch",
                "write job description for software engineer",
                "prepare financial forecast presentation"
            ],
            'CONVERSATIONAL_ADVICE': [
                "what should I do about relationship problems",
                "how to deal with stress at work",
                "give me advice on career change",
                "help me choose between two job offers",
                "personal recommendations for healthy lifestyle",
                "what's your opinion on this situation"
            ]
        }
        
        try:
            for category, examples in category_examples.items():
                logger.info(f"Computing embeddings for category '{category}' using {len(examples)} examples")
                
                # Encode all examples for this category
                embeddings = []
                for example in examples:
                    embedding = self.embedding_model.encode(example)
                    embeddings.append(embedding)
                
                # Use mean of all embeddings as category representation
                category_embedding = np.mean(embeddings, axis=0)
                self.category_embeddings[category] = category_embedding
                
                logger.info(f"Category '{category}' embedding computed, shape: {category_embedding.shape}")
                
        except Exception as e:
            logger.warning(f"Could not compute enhanced category embeddings: {e}")

    def _classify_by_embedding(self, processed_query: ProcessedQuery) -> Tuple[str, float]:
        """Enhanced embedding-based classification with better similarity computation"""
        logger.info("Starting enhanced embedding-based classification")
        
        if not self.embedding_model or not self.category_embeddings:
            logger.warning("Embedding model or category embeddings not available")
            return self._classify_by_rules(processed_query)
        
        try:
            # Use both normalized query and original for better matching
            queries_to_test = [
                processed_query.normalized_query,
                processed_query.original_query
            ]
            
            all_similarities = {}
            
            for query_text in queries_to_test:
                logger.info(f"Computing embedding for: '{query_text[:100]}...'")
                query_embedding = self.embedding_model.encode(query_text)
                
                # Calculate similarities with each category
                for category, cat_embedding in self.category_embeddings.items():
                    similarity = np.dot(query_embedding, cat_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(cat_embedding)
                    )
                    
                    if category not in all_similarities:
                        all_similarities[category] = []
                    all_similarities[category].append(similarity)
            
            # Take the maximum similarity across all query variants
            final_similarities = {}
            for category, similarities in all_similarities.items():
                final_similarities[category] = max(similarities)
                logger.info(f"Best similarity with '{category}': {final_similarities[category]:.4f}")
            
            best_category = max(final_similarities, key=final_similarities.get)
            confidence = float(final_similarities[best_category])
            
            # Apply confidence boost for very high similarities
            if confidence > 0.8:
                confidence = min(confidence + 0.1, 1.0)
                logger.info(f"High similarity confidence boost applied")
            
            logger.info(f"Embedding classification result: '{best_category}' with confidence {confidence:.4f}")
            
            return best_category, confidence
            
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
            return self._classify_by_rules(processed_query)
    def _classify_by_rules(self, processed_query: ProcessedQuery) -> Tuple[str, float]:
        """Enhanced rule-based classification with better matching and scoring"""
        import re
        
        logger.info("Starting enhanced rule-based classification")
        query_text = processed_query.normalized_query.lower()
        original_text = processed_query.original_query.lower()
        
        logger.info(f"Classifying query: '{query_text[:100]}...'")
        
        # Enhanced category patterns with weights and synonyms
        enhanced_patterns = {
            'CODE_TECHNICAL': {
                'primary': ['python', 'javascript', 'java', 'code', 'programming', 'debug', 'api', 'database', 'algorithm', 'function'],
                'secondary': ['coding', 'software', 'development', 'bug', 'error', 'compile', 'syntax', 'variable', 'class', 'method'],
                'phrases': ['write.*function', 'debug.*code', 'fix.*error', 'implement.*algorithm', 'create.*api']
            },
            'MATHEMATICAL_SCIENTIFIC': {
                'primary': ['calculate', 'mathematics', 'equation', 'formula', 'statistics', 'analyze', 'research', 'study'],
                'secondary': ['data analysis', 'correlation', 'hypothesis', 'methodology', 'scientific', 'numerical'],
                'phrases': ['benefits.*of.*renewable', 'economic.*impact', 'analyze.*data', 'research.*on']
            },
            'EDUCATIONAL_ACADEMIC': {
                'primary': ['explain', 'teach', 'learn', 'education', 'academic', 'study'],
                'secondary': ['benefits of', 'advantages of', 'what are', 'how does', 'why is', 'causes of'],
                'phrases': ['explain.*concept', 'teach.*me', 'help.*understand', 'benefits.*of.*\\w+']
            },
            'CREATIVE_ARTISTIC': {
                'primary': ['write story', 'poem', 'creative', 'art', 'music', 'design'],
                'secondary': ['narrative', 'character', 'plot', 'artistic', 'compose', 'brainstorm'],
                'phrases': ['write.*story', 'create.*poem', 'design.*logo', 'compose.*lyrics']
            },
            'BUSINESS_PROFESSIONAL': {
                'primary': ['business', 'marketing', 'strategy', 'professional', 'proposal'],
                'secondary': ['client', 'project', 'budget', 'revenue', 'market', 'financial'],
                'phrases': ['business.*plan', 'marketing.*strategy', 'professional.*email']
            },
            'CONVERSATIONAL_ADVICE': {
                'primary': ['advice', 'help me', 'what should i', 'recommend', 'opinion'],
                'secondary': ['personal', 'relationship', 'lifestyle', 'suggest', 'guidance'],
                'phrases': ['what should i do', 'give.*advice', 'help.*with.*problem']
            }
        }
        
        category_scores = {}
        
        for category, patterns in enhanced_patterns.items():
            total_score = 0
            matches_found = []
            
            # Primary keywords (weight: 3)
            for keyword in patterns['primary']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_text):
                    total_score += 3
                    matches_found.append(f"PRIMARY: {keyword}")
            
            # Secondary keywords (weight: 1)
            for keyword in patterns['secondary']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_text):
                    total_score += 1
                    matches_found.append(f"SECONDARY: {keyword}")
            
            # Phrase patterns (weight: 5)
            for phrase_pattern in patterns['phrases']:
                if re.search(phrase_pattern, query_text):
                    total_score += 5
                    matches_found.append(f"PHRASE: {phrase_pattern}")
            
            # Calculate normalized score
            max_possible_score = len(patterns['primary']) * 3 + len(patterns['secondary']) * 1 + len(patterns['phrases']) * 5
            normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0
            
            category_scores[category] = {
                'raw_score': total_score,
                'normalized_score': normalized_score,
                'matches': matches_found,
                'max_possible': max_possible_score
            }
            
            if matches_found:
                logger.info(f"Category '{category}': {total_score}/{max_possible_score} points ({normalized_score:.3f}) - {matches_found}")
        
        # Find best category
        if not category_scores or all(score['raw_score'] == 0 for score in category_scores.values()):
            logger.warning("No keyword matches found, using fallback")
            return 'CONVERSATIONAL_ADVICE', 0.05
        
        best_category = max(category_scores, key=lambda x: category_scores[x]['normalized_score'])
        best_score = category_scores[best_category]['normalized_score']
        
        # Better confidence calculation
        confidence = min(best_score * 1.5, 1.0)  # Scale up but cap at 1.0
        
        # Bonus for multiple strong matches
        if category_scores[best_category]['raw_score'] >= 8:
            confidence = min(confidence + 0.2, 1.0)
            logger.info("Multiple strong matches bonus applied")
        
        logger.info(f"Enhanced rule-based result: '{best_category}' with confidence {confidence:.3f}")
        logger.info(f"Raw score: {category_scores[best_category]['raw_score']}, Normalized: {best_score:.3f}")
        
        return best_category, confidence
    
    def _classify_by_llm(self, processed_query: ProcessedQuery) -> Tuple[str, float]:
        """
        LLM-based classification using structured prompting
        """
        logger.info("Starting LLM-based classification")
        
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to embedding classification")
            return self._classify_by_embedding(processed_query)
        
        # Define categories with detailed descriptions for the LLM
        category_definitions = {
            'CODE_TECHNICAL': 'Programming, software development, debugging, algorithms, APIs, databases, coding problems, technical implementation',
            'MATHEMATICAL_SCIENTIFIC': 'Mathematics, calculations, scientific analysis, data analysis, research, statistics, scientific methodology, economic analysis',
            'EDUCATIONAL_ACADEMIC': 'Learning, teaching, explanations, academic topics, educational content, study help, concept explanations',
            'CREATIVE_ARTISTIC': 'Creative writing, stories, poems, art, music, design, artistic expression, creative brainstorming',
            'BUSINESS_PROFESSIONAL': 'Business strategy, marketing, professional communication, proposals, financial planning, corporate tasks, legal matters',
            'CONVERSATIONAL_ADVICE': 'Personal advice, guidance, recommendations, opinion requests, lifestyle help, relationship advice'
        }
        
        # Simplified classification prompt to avoid truncation
        classification_prompt = f"""Classify this query into ONE category:

    Query: "{processed_query.original_query}"

    Categories:
    1. CODE_TECHNICAL - Programming/coding
    2. MATHEMATICAL_SCIENTIFIC - Math/science/analysis  
    3. EDUCATIONAL_ACADEMIC - Learning/teaching/explanations
    4. CREATIVE_ARTISTIC - Creative writing/art/design
    5. BUSINESS_PROFESSIONAL - Business/legal/professional
    6. CONVERSATIONAL_ADVICE - Personal advice/opinions

    Rules:
    - Legal questions (lawsuits, contracts) = BUSINESS_PROFESSIONAL
    - "What are benefits of..." = EDUCATIONAL_ACADEMIC
    - Programming help = CODE_TECHNICAL
    - Personal advice = CONVERSATIONAL_ADVICE

    Respond with JSON only. Be conservative with confidence (0.0-1.0):
    {{"category": "CATEGORY_NAME", "confidence": 0.85, "reasoning": "Brief explanation of why this category was chosen"}}"""

        try:
            logger.info("Sending classification request to LLM...")
            logger.info(f"Query to classify: '{processed_query.original_query}'")
            
            response = self.llm_client.chat.completions.create(
                model=self.classification_model,
                messages=[{
                    "role": "system", 
                    "content": "Respond only with JSON. No explanations, no thinking, just JSON."
                }, {
                    "role": "user", 
                    "content": classification_prompt
                }],
                temperature=0.0,  # Zero temperature for consistency
                max_tokens=100    # Short response to avoid truncation
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"LLM classification response received: '{response_text}'")
            
            # Clean and parse JSON
            try:
                # Remove any non-JSON content
                cleaned_response = response_text
                
                # Remove thinking tags if present
                if '<think>' in cleaned_response:
                    think_end = cleaned_response.find('</think>')
                    if think_end != -1:
                        cleaned_response = cleaned_response[think_end + 8:].strip()
                
                # Extract JSON
                import re
                json_match = re.search(r'\{[^}]*\}', cleaned_response)
                if json_match:
                    json_text = json_match.group()
                else:
                    json_text = cleaned_response
                
                logger.info(f"JSON to parse: '{json_text}'")
                
                # Parse JSON
                import json
                classification_result = json.loads(json_text)
                
                category = classification_result.get('category', '').upper()
                confidence = float(classification_result.get('confidence', 0.7))
                
                # Validate category
                valid_categories = list(category_definitions.keys())
                if category not in valid_categories:
                    logger.warning(f"Invalid category '{category}', trying to map it")
                    # Try to map partial matches
                    for valid_cat in valid_categories:
                        if category in valid_cat or valid_cat in category:
                            category = valid_cat
                            break
                    else:
                        logger.warning(f"Could not map category, defaulting to BUSINESS_PROFESSIONAL")
                        category = 'BUSINESS_PROFESSIONAL'
                
                # Validate confidence
                confidence = max(0.0, min(1.0, confidence))
                
                logger.info(f"LLM classification successful: {category} (confidence: {confidence:.4f})")
                return category, confidence
                
            except (json.JSONDecodeError, ValueError, KeyError) as parse_error:
                logger.warning(f"JSON parsing failed: {parse_error}")
                logger.warning(f"Response was: '{response_text}'")
                
                # Manual extraction fallback
                category_match = re.search(r'"category":\s*"([^"]+)"', response_text, re.IGNORECASE)
                confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response_text)
                
                if category_match:
                    category = category_match.group(1).upper()
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                    
                    # Map to valid category
                    if 'BUSINESS' in category or 'PROFESSIONAL' in category:
                        category = 'BUSINESS_PROFESSIONAL'
                    elif 'EDUCATIONAL' in category or 'ACADEMIC' in category:
                        category = 'EDUCATIONAL_ACADEMIC'
                    elif 'CODE' in category or 'TECHNICAL' in category:
                        category = 'CODE_TECHNICAL'
                    elif 'CONVERSATIONAL' in category or 'ADVICE' in category:
                        category = 'CONVERSATIONAL_ADVICE'
                    elif 'CREATIVE' in category or 'ARTISTIC' in category:
                        category = 'CREATIVE_ARTISTIC'
                    elif 'MATHEMATICAL' in category or 'SCIENTIFIC' in category:
                        category = 'MATHEMATICAL_SCIENTIFIC'
                    else:
                        category = 'BUSINESS_PROFESSIONAL'  # Default for legal questions
                    
                    logger.info(f"Manual extraction successful: {category} (confidence: {confidence:.4f})")
                    return category, confidence
                
                # Final fallback - for lawsuit questions, default to BUSINESS_PROFESSIONAL
                if 'lawsuit' in processed_query.original_query.lower() or 'legal' in processed_query.original_query.lower():
                    logger.info("Detected legal question, defaulting to BUSINESS_PROFESSIONAL")
                    return 'BUSINESS_PROFESSIONAL', 0.8
                
                logger.warning("All parsing failed, falling back to embedding classification")
                return self._classify_by_embedding(processed_query)
                
        except Exception as e:
            logger.error(f"LLM classification request failed: {str(e)}")
            return self._classify_by_embedding(processed_query)
    
    def _determine_complexity(self, processed_query: ProcessedQuery) -> str:
        """
        Determine query complexity level based on features
        
        Args:
            processed_query: Preprocessed query object
            
        Returns:
            Complexity level string
        """
        logger.info("Determining query complexity level")
        features = processed_query.features
        word_count = features.get('word_count', 0)
        complexity_features = features.get('complexity', {})
        
        logger.info(f"Word count: {word_count}")
        logger.info(f"Complexity features: {complexity_features}")
        
        # Simple heuristic for complexity
        if word_count < 5:
            complexity = 'BASIC'
            logger.info(f"Complexity determined as {complexity} due to low word count ({word_count})")
        elif word_count < 20:
            if complexity_features.get('has_code_markers') or \
               complexity_features.get('has_math_symbols'):
                complexity = 'INTERMEDIATE'
                logger.info(f"Complexity determined as {complexity} due to technical markers")
            else:
                complexity = 'BASIC'
                logger.info(f"Complexity determined as {complexity} due to moderate word count without technical markers")
        else:
            complexity = 'ADVANCED'
            logger.info(f"Complexity determined as {complexity} due to high word count ({word_count})")
        
        return complexity
    
    def _determine_response_format(self, processed_query: ProcessedQuery, category: str) -> str:
        """
        Determine expected response format based on query and category
        
        Args:
            processed_query: Preprocessed query object
            category: Classified category
            
        Returns:
            Response format string
        """
        logger.info(f"Determining response format for category '{category}'")
        features = processed_query.features
        complexity_features = features.get('complexity', {})
        
        # Check for explicit format requests
        if complexity_features.get('has_code_markers'):
            response_format = 'CODE'
            logger.info(f"Response format determined as {response_format} due to code markers")
            return response_format
        
        # Category-based format prediction
        format_mapping = {
            'CODE_TECHNICAL': 'CODE',
            'MATHEMATICAL_SCIENTIFIC': 'STRUCTURED',
            'CREATIVE_ARTISTIC': 'NARRATIVE',
            'BUSINESS_PROFESSIONAL': 'STRUCTURED',
            'EDUCATIONAL_ACADEMIC': 'NARRATIVE',
            'CONVERSATIONAL_ADVICE': 'INTERACTIVE'
        }
        
        response_format = format_mapping.get(category, 'NARRATIVE')
        logger.info(f"Response format determined as {response_format} based on category mapping")
        
        return response_format
    
    def classify(self, processed_query: ProcessedQuery) -> QueryClassification:
        """
        Main classification method combining LLM, embedding, and rule-based approaches
        
        Args:
            processed_query: Preprocessed query object
            
        Returns:
            QueryClassification object with results
        """
        logger.info("Starting enhanced classification pipeline with LLM")
        logger.info(f"Query to classify: '{processed_query.original_query[:100]}...'")
        
        # Try LLM-based classification first (if available)
        if self.llm_client:
            logger.info("Phase 1: LLM-based classification")
            llm_category, llm_confidence = self._classify_by_llm(processed_query)
            logger.info(f"LLM result: {llm_category} (confidence: {llm_confidence:.4f})")
        else:
            llm_category, llm_confidence = None, 0.0
            logger.info("Phase 1: LLM-based classification - SKIPPED (no LLM client)")
        
        # Try embedding-based classification
        logger.info("Phase 2: Embedding-based classification")
        embedding_category, embedding_confidence = self._classify_by_embedding(processed_query)
        logger.info(f"Embedding result: {embedding_category} (confidence: {embedding_confidence:.4f})")
        
        # Try rule-based classification
        logger.info("Phase 3: Rule-based classification")
        rule_category, rule_confidence = self._classify_by_rules(processed_query)
        logger.info(f"Rule-based result: {rule_category} (confidence: {rule_confidence:.4f})")
        
        # Combine results with weighted confidence
        logger.info("Phase 4: Combining classification results")
        
        # Define confidence thresholds and weights
        high_confidence_threshold = 0.8
        medium_confidence_threshold = 0.6
        
        # LLM gets highest priority if available and confident
        if llm_category and llm_confidence >= high_confidence_threshold:
            primary_category, confidence = llm_category, llm_confidence
            method_used = "LLM (high confidence)"
            logger.info(f"Selected LLM result: {primary_category} (high confidence)")
            
        elif llm_category and llm_confidence >= medium_confidence_threshold:
            # LLM is moderately confident, check if others agree
            if embedding_category == llm_category or rule_category == llm_category:
                primary_category, confidence = llm_category, min(llm_confidence + 0.1, 1.0)
                method_used = "LLM (confirmed by other methods)"
                logger.info(f"Selected LLM result with confirmation: {primary_category}")
            else:
                primary_category, confidence = llm_category, llm_confidence
                method_used = "LLM (medium confidence)"
                logger.info(f"Selected LLM result: {primary_category} (medium confidence)")
                
        elif embedding_confidence >= high_confidence_threshold:
            primary_category, confidence = embedding_category, embedding_confidence
            method_used = "Embedding (high confidence)"
            logger.info(f"Selected embedding result: {primary_category} (LLM not confident enough)")
            
        elif rule_confidence >= high_confidence_threshold:
            primary_category, confidence = rule_category, rule_confidence
            method_used = "Rule-based (high confidence)"
            logger.info(f"Selected rule-based result: {primary_category} (others not confident)")
            
        elif llm_category and llm_confidence > max(embedding_confidence, rule_confidence):
            primary_category, confidence = llm_category, llm_confidence
            method_used = "LLM (best available)"
            logger.info(f"Selected LLM result: {primary_category} (best of low confidence options)")
            
        elif embedding_confidence > rule_confidence:
            primary_category, confidence = embedding_category, embedding_confidence
            method_used = "Embedding (best non-LLM)"
            logger.info(f"Selected embedding result: {primary_category}")
            
        else:
            primary_category, confidence = rule_category, rule_confidence
            method_used = "Rule-based (fallback)"
            logger.info(f"Selected rule-based result: {primary_category}")
        
        # Determine other attributes
        logger.info("Phase 5: Determining complexity and response format")
        complexity = self._determine_complexity(processed_query)
        response_format = self._determine_response_format(processed_query, primary_category)
        
        # Create secondary categories from other high-confidence results
        secondary_categories = []
        for cat, conf in [(embedding_category, embedding_confidence), (rule_category, rule_confidence)]:
            if cat != primary_category and conf >= 0.6:
                secondary_categories.append(cat)
        
        classification_result = QueryClassification(
            primary_category=primary_category,
            confidence=confidence,
            secondary_categories=secondary_categories,
            complexity_level=complexity,
            response_format=response_format
        )
        
        logger.info("Classification completed successfully")
        logger.info(f"Final result - Category: {primary_category}, Confidence: {confidence:.4f}, Method: {method_used}")
        logger.info(f"Secondary categories: {secondary_categories}")
        
        return classification_result


class LLMRouter:
    """
    Routes queries to appropriate LLMs based on classification results
    """
    
    def __init__(self, api_key_path: str):
        """
        Args:
            api_key_path: Path to file containing Nebius API key
        """
        logger.info(f"Initializing LLMRouter with API key from: {api_key_path}")
        
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
            
            self.client = OpenAI(
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=api_key,
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
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
        
        logger.info(f"Specialized models configured: {self.specialized_models}")
        logger.info(f"Default model: {self.default_model}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info("LLMRouter initialization completed")
    
    def _select_model(self, classification: QueryClassification) -> str:
        """
        Select appropriate model based on classification
        
        Args:
            classification: Query classification results
            
        Returns:
            Model name to use
        """
        logger.info(f"Selecting model for category: {classification.primary_category}, confidence: {classification.confidence:.4f}")
        
        if classification.confidence >= self.confidence_threshold:
            selected_model = self.specialized_models.get(
                classification.primary_category, 
                self.default_model
            )
            logger.info(f"High confidence classification - selected specialized model: {selected_model}")
        else:
            selected_model = self.default_model
            logger.info(f"Low confidence classification - selected default model: {selected_model}")
        
        return selected_model
    
    def _build_system_prompt(self, classification: QueryClassification) -> str:
        """
        Build system prompt based on classification results
        
        Args:
            classification: Query classification results
            
        Returns:
            System prompt string
        """
        logger.info(f"Building system prompt for category: {classification.primary_category}")
        
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
            logger.info(f"Added category-specific instruction: {category_instruction}")
        
        # Add complexity-based instructions
        if classification.complexity_level == 'BASIC':
            complexity_instruction = " Keep your explanation simple and beginner-friendly."
            base_prompt += complexity_instruction
            logger.info(f"Added basic complexity instruction: {complexity_instruction}")
        elif classification.complexity_level == 'ADVANCED':
            complexity_instruction = " Provide detailed, comprehensive analysis suitable for experts."
            base_prompt += complexity_instruction
            logger.info(f"Added advanced complexity instruction: {complexity_instruction}")
        
        logger.info(f"Final system prompt built: '{base_prompt[:100]}...'")
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
        logger.info(f"Starting response generation for query: '{query[:100]}...'")
        
        model = self._select_model(classification)
        system_prompt = self._build_system_prompt(classification)
        
        logger.info(f"Using model: {model}")
        logger.info(f"System prompt length: {len(system_prompt)} characters")
        
        try:
            logger.info("Sending request to LLM API...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            generated_response = response.choices[0].message.content
            logger.info(f"Response generated successfully, length: {len(generated_response)} characters")
            logger.info(f"Response preview: '{generated_response[:200]}...'")
            
            return generated_response
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error generating a response: {str(e)}"
            logger.error(f"Error generating response: {str(e)}")
            return error_message


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
        logger.info(f"Initializing QueryJudge with model: {model_name}")
        self.model_name = model_name
        
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
            
            self.client = OpenAI(
                base_url="https://api.studio.nebius.ai/v1/",
                api_key=api_key,
            )
            logger.info("Judge client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize judge client: {e}")
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
        logger.info(f"Starting response judgment for query: '{query[:50]}...'")
        logger.info(f"Response to judge (length: {len(response)}): '{response[:100]}...'")
        
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
            logger.info("Sending judgment request to LLM API...")
            judge_api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            
            # Extract the response text
            response_text = judge_api_response.choices[0].message.content
            logger.info(f"Judge response received (length: {len(response_text)}): '{response_text[:200]}...'")
            
            # Clean the response text - remove any markdown formatting or extra text
            cleaned_response = response_text.strip()
            
            # Look for JSON content between curly braces if there's extra text
            import re
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                logger.info(f"Extracted JSON text: '{json_text[:200]}...'")
            else:
                json_text = cleaned_response
                logger.info("No JSON brackets found, using full response as JSON")
            
            try:
                judgment = json.loads(json_text)
                logger.info(f"Judgment parsed successfully: Score={judgment.get('score')}, Correct={judgment.get('is_correct')}")
                
                # Validate the parsed judgment has required fields
                if 'score' not in judgment or 'justification' not in judgment:
                    logger.warning(f"Missing required fields in judgment: {judgment}")
                    raise ValueError("Missing required fields")
                    
                return judgment
                
            except (json.JSONDecodeError, ValueError) as json_error:
                logger.warning(f"Could not parse judge response as JSON: {json_error}")
                logger.warning(f"Raw response: {response_text}")
                logger.warning(f"Cleaned response: {cleaned_response}")
                logger.warning(f"JSON text attempted: {json_text}")
                
                # Try to extract score manually as fallback
                score_match = re.search(r'"score":\s*(\d+)', response_text)
                if score_match:
                    fallback_score = int(score_match.group(1))
                    logger.info(f"Extracted score manually: {fallback_score}")
                    return {
                        "score": fallback_score,
                        "justification": f"Manually extracted from response: {response_text[:200]}",
                        "is_correct": True
                    }
                
                # Final fallback
                return {
                    "score": 3,
                    "justification": f"Could not parse judge response. Raw response: {response_text[:200]}",
                    "is_correct": True
                }
                
        except Exception as e:
            logger.error(f"Error in judging: {str(e)}")
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
        logger.info("Initializing LLMQueryRouter - main orchestrator")
        logger.info(f"API key path: {api_key_path}")
        
        try:
            logger.info("Initializing QueryPreprocessor...")
            self.preprocessor = QueryPreprocessor()
            
            logger.info("Initializing QueryClassifier with LLM support...")
            self.classifier = QueryClassifier()  # Pass API key
            
            logger.info("Initializing LLMRouter...")
            self.router = LLMRouter(api_key_path)
            
            logger.info("Initializing QueryJudge...")
            self.judge = QueryJudge(api_key_path)
            
            logger.info("LLMQueryRouter initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMQueryRouter: {e}")
            raise
    
    def process_query(self, query: str, include_judgment: bool = False) -> Dict:
        """
        Main pipeline: preprocess -> classify -> route -> generate -> (judge)
        
        Args:
            query: Raw user query
            include_judgment: Whether to include quality judgment
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info("="*80)
        logger.info("STARTING QUERY PROCESSING PIPELINE")
        logger.info("="*80)
        logger.info(f"Input query: '{query}'")
        logger.info(f"Include judgment: {include_judgment}")
        
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            logger.info("\n" + "="*50)
            logger.info("STEP 1: QUERY PREPROCESSING")
            logger.info("="*50)
            processed_query = self.preprocessor.process(query)
            preprocessing_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {preprocessing_time:.3f} seconds")
            
            # Step 2: Classify query
            logger.info("\n" + "="*50)
            logger.info("STEP 2: QUERY CLASSIFICATION")
            logger.info("="*50)
            classification_start = time.time()
            classification = self.classifier.classify(processed_query)
            classification_time = time.time() - classification_start
            logger.info(f"Classification completed in {classification_time:.3f} seconds")
            
            # Step 3: Generate response
            logger.info("\n" + "="*50)
            logger.info("STEP 3: RESPONSE GENERATION")
            logger.info("="*50)
            generation_start = time.time()
            response = self.router.generate_response(query, classification)
            generation_time = time.time() - generation_start
            logger.info(f"Response generation completed in {generation_time:.3f} seconds")
            
            # Step 4: Optional judgment
            judgment = None
            judgment_time = 0
            if include_judgment:
                logger.info("\n" + "="*50)
                logger.info("STEP 4: RESPONSE JUDGMENT")
                logger.info("="*50)
                judgment_start = time.time()
                judgment = self.judge.judge_response(query, response)
                judgment_time = time.time() - judgment_start
                logger.info(f"Response judgment completed in {judgment_time:.3f} seconds")
            else:
                logger.info("\n" + "="*50)
                logger.info("STEP 4: RESPONSE JUDGMENT - SKIPPED")
                logger.info("="*50)
            
            total_processing_time = time.time() - start_time
            
            result = {
                'response': response,
                'classification': {
                    'category': classification.primary_category,
                    'confidence': classification.confidence,
                    'complexity': classification.complexity_level,
                    'format': classification.response_format
                },
                'judgment': judgment,
                'processing_time_seconds': total_processing_time,
                'timing_breakdown': {
                    'preprocessing': preprocessing_time,
                    'classification': classification_time,
                    'generation': generation_time,
                    'judgment': judgment_time
                },
                'success': True
            }
            
            logger.info("\n" + "="*80)
            logger.info("QUERY PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_processing_time:.3f} seconds")
            logger.info(f"Final classification: {classification.primary_category} (confidence: {classification.confidence:.4f})")
            logger.info(f"Response length: {len(response)} characters")
            if judgment:
                logger.info(f"Judgment score: {judgment.get('score', 'N/A')}/5")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error("="*80)
            logger.error("QUERY PROCESSING PIPELINE FAILED")
            logger.error("="*80)
            logger.error(f"Error occurred after {error_time:.3f} seconds: {str(e)}")
            logger.error("Returning error response")
            
            return {
                'response': f"I'm sorry, I encountered an error processing your query: {str(e)}",
                'error': str(e),
                'processing_time_seconds': error_time,
                'success': False
            }


# Example usage
if __name__ == "__main__":
    logger.info("Starting main execution")
    
    # Initialize the router
    logger.info("Initializing LLMQueryRouter...")
    router = LLMQueryRouter("Nebius_api_key.txt")
    
    # Process a sample query
    test_query = "If I want to file a lawsuit against a company, what steps should I take?"
    logger.info(f"Processing test query: '{test_query}'")
    
    result = router.process_query(test_query, include_judgment=False)
    
    print(f"Query: {test_query}")
    print(f"Category: {result['classification']['category']}")
    print(f"Confidence: {result['classification']['confidence']:.2f}")
    print(f"Response: {result['response']}")
    
    if result.get('judgment'):
        print(f"Quality Score: {result['judgment']['score']}/5")
        print(f"Justification: {result['judgment']['justification']}")
    
    logger.info("Main execution completed")