# filepath: c:\Users\maorb\work\Tryaii\classify_query.py
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import re
import numpy as np
from collections import defaultdict, Counter
# Configure concise logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Enhanced imports for NLP processing
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    
    # Download required NLTK data if not present
    def ensure_nltk_data():
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'maxent_ne_chunker', 'words']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data_name}')
                    except LookupError:
                        try:
                            nltk.data.find(f'chunkers/{data_name}')
                        except LookupError:
                            logger.info(f"Downloading NLTK data: {data_name}")
                            nltk.download(data_name, quiet=True)
    
    # Initialize NLTK data
    ensure_nltk_data()
    NLTK_AVAILABLE = True
    logger.info("NLTK successfully loaded and configured")
    
except ImportError:
    logger.warning("NLTK not available - falling back to basic text processing")
    NLTK_AVAILABLE = False


# Enhanced NLP Features for Advanced Classification
from collections import defaultdict
from typing import Set
try:
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from nltk.corpus import wordnet
    ADVANCED_NLTK_AVAILABLE = True
    logger.info("Advanced NLTK features available")
except ImportError:
    ADVANCED_NLTK_AVAILABLE = False
    logger.warning("Advanced NLTK features not available")

class DomainLexiconBuilder:
    """Builds comprehensive domain-specific lexicons using NLTK and external resources"""
    
    def __init__(self):
        if ADVANCED_NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = None
            self.lemmatizer = None
        
        # Enhanced base domain lexicons
        self.base_lexicons = {
            'technical': {
                'programming': [
                    'algorithm', 'array', 'binary', 'class', 'compile', 'debug', 'function',
                    'inheritance', 'loop', 'method', 'object', 'parameter', 'recursion',
                    'syntax', 'variable', 'framework', 'library', 'module', 'package',
                    'repository', 'version', 'branch', 'commit', 'merge', 'pull', 'push'
                ],
                'data_science': [
                    'correlation', 'regression', 'classification', 'clustering', 'feature',
                    'dataset', 'training', 'validation', 'testing', 'model', 'prediction',
                    'accuracy', 'precision', 'recall', 'statistics', 'probability',
                    # Computer vision and biomedical terms
                    'computer', 'vision', 'image', 'processing', 'segmentation', 'detection',
                    'iou', 'intersection', 'union', 'bounding', 'box', 'mask', 'pixel',
                    'opencv', 'convolution', 'filter', 'edge', 'contour', 'morphology',
                    'biomedical', 'microscopy', 'cell', 'tissue', 'medical', 'imaging',
                    'analysis', 'measurement', 'quantification', 'visualization', 'optimize'
                ]
            },
            'business': {
                'legal': [
                    'contract', 'agreement', 'clause', 'liability', 'negligence',
                    'compliance', 'regulation', 'statute', 'lawsuit', 'litigation',
                    'intellectual', 'property', 'trademark', 'copyright', 'patent'
                ],
                'finance': [
                    'revenue', 'profit', 'budget', 'investment', 'return', 'capital',
                    'expense', 'income', 'cash', 'flow', 'balance', 'sheet'
                ]
            },
            'academic': {
                'research': [
                    'hypothesis', 'methodology', 'experiment', 'observation', 'analysis',
                    'conclusion', 'literature', 'review', 'citation', 'benefits', 'advantages'
                ],
                'science': [
                    'theory', 'principle', 'law', 'equation', 'formula', 'calculation',
                    'renewable', 'energy', 'climate', 'environment', 'sustainability'
                ]
            }
        }
        
        self.expanded_lexicons = self._expand_lexicons()
    
    def _expand_lexicons(self) -> Dict[str, Set[str]]:
        """Expand base lexicons using WordNet and morphological variations"""
        expanded = defaultdict(set)
        
        for domain, subdomains in self.base_lexicons.items():
            for subdomain, terms in subdomains.items():
                key = f"{domain}_{subdomain}"
                expanded[key] = set()
                
                for term in terms:
                    # Add original term
                    expanded[key].add(term)
                    
                    # Add morphological variations
                    expanded[key].add(term + 's')  # plural
                    expanded[key].add(term + 'ing')  # gerund
                    expanded[key].add(term + 'ed')  # past tense
                    
                    # Add WordNet synsets if available
                    if ADVANCED_NLTK_AVAILABLE:
                        try:
                            synsets = wordnet.synsets(term)
                            for synset in synsets[:2]:  # Limit to top 2 synsets
                                for lemma in synset.lemmas():
                                    synonym = lemma.name().replace('_', ' ')
                                    if len(synonym) > 2:
                                        expanded[key].add(synonym)
                        except Exception:
                            pass  # Skip if WordNet lookup fails
                    
                    # Add stem and lemma if available
                    if self.stemmer and self.lemmatizer:
                        try:
                            expanded[key].add(self.stemmer.stem(term))
                            expanded[key].add(self.lemmatizer.lemmatize(term))
                        except Exception:
                            pass
        
        # Create aggregate domain lexicons
        for domain in ['technical', 'business', 'academic']:
            expanded[domain] = set()
            for key in expanded.keys():
                if key.startswith(domain + '_'):
                    expanded[domain].update(expanded[key])
        
        return dict(expanded)
    
    def get_domain_terms(self, domain: str) -> Set[str]:
        """Get all terms for a specific domain"""
        return self.expanded_lexicons.get(domain, set())

class EnhancedQueryFeatures:
    """Enhanced feature set for advanced query classification"""
    
    def __init__(self):
        self.domain_densities = {}
        self.intent_patterns = []
        self.question_type = None
        self.complexity_indicators = {}
        self.lexical_diversity = 0.0
        self.pos_tags = []
        self.named_entities = []


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
        
        # Common action verbs that indicate query intent
        self.action_verbs = [
            'write', 'create', 'build', 'generate', 'explain', 'analyze',
            'debug', 'fix', 'optimize', 'compare', 'summarize', 'teach',
            'implement', 'design', 'develop', 'code', 'program', 'solve',
            'help', 'show', 'demonstrate', 'convert', 'transform', 'migrate'
        ]
        
        logger.info("QueryPreprocessor ready")
    
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
        logger.info(f"Normalizing: '{query[:30]}...'")
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
        Enhanced feature extraction with advanced NLP analysis
        
        Args:
            query: Normalized query string
            
        Returns:
            Dictionary containing extracted features with enhanced NLP analysis
        """
        logger.info(f"Extracting enhanced features from: '{query[:30]}...'")
        import re
        
        # Initialize lexicon builder if not exists
        if not hasattr(self, 'lexicon_builder'):
            self.lexicon_builder = DomainLexiconBuilder()
        
        words = query.lower().split()
        
        # Enhanced technical keyword extraction
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
        
        # Enhanced domain-specific term extraction
        domain_terms = {}
        for domain in ['technical', 'business', 'academic']:
            domain_lexicon = self.lexicon_builder.get_domain_terms(domain)
            found_terms = [word for word in words if word in domain_lexicon]
            # Add multi-word terms
            for term in domain_lexicon:
                if ' ' in term and term in query_lower and term not in found_terms:
                    found_terms.append(term)
            domain_terms[domain] = list(set(found_terms))
        
        # Find action verbs
        action_verbs = [word for word in words if word in self.action_verbs]
        
        # Enhanced complexity indicators
        complexity_indicators = {
            'has_code_markers': bool(re.search(r'[{}()\[\];]|def\s+|class\s+|import\s+', query)),
            'has_math_symbols': bool(re.search(r'[∑∏∫∂∇]|\\|equation|formula', query)),
            'has_technical_jargon': len([w for w in words if w in ['implementation', 'architecture', 'optimization', 'scalability', 'performance']]) > 0,
            'has_academic_language': len([w for w in words if w in ['hypothesis', 'methodology', 'empirical', 'theoretical', 'systematic']]) > 0,
            'has_formal_language': len([w for w in words if w in ['furthermore', 'consequently', 'nevertheless', 'accordingly']]) > 0,
            'has_special_chars': bool(re.search(r'[<>@#$%^&*]', query)),
            'word_count': len(words),
            'char_count': len(query),
            'has_numbers': bool(re.search(r'\d', query)),
            'has_urls': bool(re.search(r'http[s]?://|www\.', query)),
            'question_words': len([w for w in words if w in ['what', 'how', 'why', 'when', 'where', 'which']]),
            'tech_density': len(tech_keywords) / len(words) if words else 0
        }
        
        # Intent pattern detection
        intent_patterns = {
            'how_to': bool(re.search(r'\b(how\s+to|how\s+do\s+i|how\s+can\s+i)\b', query)),
            'what_is': bool(re.search(r'\b(what\s+is|what\s+are|what\s+does)\b', query)),
            'why': bool(re.search(r'\b(why\s+is|why\s+do|why\s+does|why\s+are)\b', query)),
            'explain': bool(re.search(r'\b(explain|describe|tell\s+me\s+about)\b', query)),
            'create': bool(re.search(r'\b(create|build|make|generate|write)\b', query)),
            'fix': bool(re.search(r'\b(fix|debug|solve|resolve|troubleshoot)\b', query)),
            'analyze': bool(re.search(r'\b(analyze|examine|study|investigate)\b', query)),
            'optimize': bool(re.search(r'\b(optimize|improve|enhance|maximize|minimize)\b', query)),
            'find': bool(re.search(r'\b(find|locate|identify|detect|discover)\b', query)),
            'calculate': bool(re.search(r'\b(calculate|compute|measure|determine)\b', query))
        }
        
        # Question type detection
        question_type = 'general'
        if re.search(r'^(what|when|where|who|which)\b', query.lower().strip()):
            question_type = 'factual'
        elif re.search(r'^(how|how\s+to|how\s+do|how\s+can)\b', query.lower().strip()):
            question_type = 'procedural'
        elif re.search(r'^(why|what\s+causes|what\s+makes)\b', query.lower().strip()):
            question_type = 'causal'
        elif re.search(r'\b(better|worse|difference|compare|versus|vs)\b', query.lower()):
            question_type = 'comparative'
        elif re.search(r'\b(should|would|could|recommend|suggest|opinion)\b', query.lower()):
            question_type = 'evaluative'
        elif re.search(r'\b(create|write|design|compose|generate)\b', query.lower()):
            question_type = 'creative'
        elif re.search(r'\b(analyze|examine|study|investigate|research)\b', query.lower()):
            question_type = 'analytical'
        
        # Domain density calculation
        total_content_words = len([w for w in words if len(w) > 2])  # Filter short words
        domain_densities = {}
        for domain, terms in domain_terms.items():
            domain_densities[domain] = len(terms) / total_content_words if total_content_words > 0 else 0
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        primary_tech_category = self._categorize_tech_terms(tech_keywords)
        
        features = {
            'tech_keywords': tech_keywords,
            'domain_terms': domain_terms,
            'domain_densities': domain_densities,
            'action_verbs': action_verbs,
            'word_count': len(words),
            'complexity': complexity_indicators,
            'intent_patterns': intent_patterns,
            'question_type': question_type,
            'lexical_diversity': lexical_diversity,
            'has_tech_terms': len(tech_keywords) > 0,
            'tech_term_count': len(tech_keywords),
            'primary_tech_category': primary_tech_category,
            'has_technical_context': len(tech_keywords) > 0 or domain_densities.get('technical', 0) > 0.1,
            'has_business_context': domain_densities.get('business', 0) > 0.1,
            'has_academic_context': domain_densities.get('academic', 0) > 0.1
        }
        
        logger.info(f"Enhanced features - tech terms: {len(tech_keywords)}, domain densities: {domain_densities}")
        logger.info(f"Question type: {question_type}, Intent patterns: {[k for k, v in intent_patterns.items() if v]}")
        
        return features
        
        return features
    
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
        
        result = max(category_scores, key=category_scores.get) if category_scores else 'general'
        
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
        logger.info(f"Processing: '{query[:50]}...'")
        
        try:
            # Step 1: Normalize text
            normalized = self.normalize_text(query)
            
            # Step 2: Extract features
            features = self.extract_features(normalized)
            
            # Step 3: Extract key terms and action verbs from features
            key_terms = features.get('tech_keywords', [])
            action_verbs = features.get('action_verbs', [])
            
            processed_query = ProcessedQuery(
                original_query=query,
                normalized_query=normalized,
                key_terms=key_terms,
                action_verbs=action_verbs,
                features=features
            )
            
            logger.info(f"Processed - terms: {len(key_terms)}, verbs: {len(action_verbs)}")
            
            return processed_query
            
        except Exception as e:
            logger.warning(f"Preprocessing error: {str(e)}")
            
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
        logger.info(f"Initializing QueryClassifier")
        
        # Load sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
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
        
        # Pre-compute category embeddings if model available
        self.category_embeddings = {}
        if self.embedding_model:
            logger.info("Computing category embeddings...")
            self._compute_category_embeddings()
        else:
            logger.warning("No embedding model - using rules only")

        self.llm_client = None
        api_key_path = "Nebius_api_key.txt"  # Path to your API key file
        if api_key_path:
            try:
                with open(api_key_path, "r") as f:
                    api_key = f.read().strip()
                
                self.llm_client = OpenAI(
                    base_url="https://api.studio.nebius.ai/v1/",
                    api_key=api_key,
                )
                logger.info("LLM client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
        else:
            logger.info("No API key - LLM disabled")
        
        # Classification model for LLM
        self.classification_model = "Qwen/Qwen3-235B-A22B"  # Use the best model for classification
    

        logger.info("QueryClassifier ready")

    
    def _compute_category_embeddings(self):
        """Enhanced embedding computation with comprehensive examples and NLP augmentation"""
        logger.info("Computing category embeddings...")
        
        # Significantly expanded and diverse examples per category for better representation
        category_examples = {
            'CODE_TECHNICAL': [
                # Programming fundamentals
                "write a python function to sort a list using quicksort algorithm",
                "debug this javascript error in my React component",
                "how to implement binary search algorithm in C++",
                "optimize SQL query performance for large datasets",
                "fix compilation error in my Java program",
                "create REST API endpoint using FastAPI",
                "database schema design patterns for e-commerce",
                
                # Data Science & AI
                "build machine learning model for image classification",
                "implement neural network using TensorFlow",
                "optimize hyperparameters for deep learning model",
                "create data pipeline for ETL processing",
                "implement recommendation system using collaborative filtering",
                "analyze time series data using LSTM networks",
                "build computer vision model for object detection",
                "implement natural language processing for sentiment analysis",
                
                # Systems & DevOps
                "configure Docker container for microservice deployment",
                "implement user authentication in Node.js application",
                "troubleshoot memory leak in Python application",
                "set up CI/CD pipeline using Jenkins",
                "configure load balancer for high availability",
                "implement caching strategy using Redis",
                "optimize application performance and scalability",
                "configure cloud infrastructure on AWS",
                
                # Web Development
                "create responsive web design using CSS Grid",
                "implement state management in React with Redux",
                "build GraphQL API with Apollo Server",
                "optimize webpack bundle size for production",
                "implement server-side rendering with Next.js",
                "create progressive web app with service workers"
            ],
            'MATHEMATICAL_SCIENTIFIC': [
                # Pure Mathematics
                "calculate the derivative of exponential function",
                "solve system of linear equations using matrix methods",
                "find eigenvalues and eigenvectors of transformation matrix",
                "compute integral using numerical integration methods",
                "analyze convergence of infinite series",
                "prove mathematical theorem using induction",
                "optimize function subject to constraints using Lagrange multipliers",
                "calculate probability distribution parameters",
                
                # Statistics & Data Analysis
                "perform statistical hypothesis testing on experimental data",
                "analyze correlation between variables in dataset",
                "conduct regression analysis to predict outcomes",
                "perform A/B testing for website optimization",
                "analyze survey data using statistical methods",
                "investigate patterns in financial market data",
                "study demographic trends in population data",
                "perform time series analysis and forecasting",
                
                # Scientific Research
                "design controlled experiment to test hypothesis",
                "investigate economic benefits of renewable energy",
                "study correlation between education and income levels",
                "research methodology for social science studies",
                "analyze trends in scientific publication data",
                "evaluate effectiveness of medical treatment",
                "investigate environmental impact of industrial processes",
                "study genetic factors in disease susceptibility",
                "analyze climate change data using statistical models",
                
                # Applied Mathematics
                "model population growth using differential equations",
                "optimize resource allocation using linear programming",
                "analyze network topology using graph theory",
                "simulate physical systems using numerical methods",
                "calculate risk assessment using probability theory"
            ],
            'EDUCATIONAL_ACADEMIC': [
                # Science Education
                "explain quantum mechanics concepts in simple terms",
                "teach me about photosynthesis process in plants",
                "describe the structure of DNA and its function",
                "explain how the human brain processes information",
                "teach me about the water cycle and its importance",
                "explain the concept of natural selection and evolution",
                "describe chemical bonding in molecules",
                "explain thermodynamics principles for physics students",
                
                # History & Social Sciences
                "what are the causes of World War II",
                "explain historical significance of Renaissance period",
                "describe the impact of Industrial Revolution",
                "explain the formation of democratic governments",
                "teach me about ancient civilizations and their contributions",
                "explain the causes and effects of economic recessions",
                
                # Learning Support
                "help me understand calculus fundamentals",
                "create study guide for biology exam",
                "explain the difference between mitosis and meiosis",
                "help me prepare for physics test on electromagnetism",
                "teach me about financial literacy and budgeting",
                "explain programming concepts for beginners",
                "help me understand statistics for research",
                
                # Academic Research & Methods
                "summarize main points of research paper on climate change",
                "explain methodology used in psychological studies",
                "describe advantages of different research methods",
                "help me understand statistical analysis techniques",
                "explain the peer review process in academic publishing",
                "describe ethical considerations in research",
                "help me write literature review for thesis",
                "explain how to conduct systematic review",
                
                # Benefits & Advantages Queries
                "what are the benefits of renewable energy sources",
                "explain advantages of online learning platforms",
                "describe benefits of multilingual education",
                "what are advantages of collaborative learning",
                "explain benefits of critical thinking skills"
            ],
            'CREATIVE_ARTISTIC': [
                # Creative Writing
                "write a short story about time travel paradox",
                "create a poem about autumn leaves falling",
                "develop character backstory for fantasy novel",
                "write dialogue for dramatic theater scene",
                "create compelling opening paragraph for mystery story",
                "develop plot outline for science fiction story",
                "write descriptive passage about mountain landscape",
                "create rhyming poem about friendship and loyalty",
                "develop unique setting for adventure story",
                "write emotional monologue for stage performance",
                
                # Visual Design & Arts
                "design logo for coffee shop brand identity",
                "create color palette for modern website",
                "develop visual identity for startup company",
                "design poster for music festival event",
                "create user interface for mobile app",
                "develop branding strategy for new product",
                "design infographic about environmental issues",
                "create artistic composition using geometric shapes",
                "design layout for magazine article",
                "create album cover for indie music band",
                
                # Creative Brainstorming
                "brainstorm creative solutions for urban transportation",
                "generate innovative ideas for team building activities",
                "develop unique concept for restaurant theme",
                "create original game mechanics for board game",
                "brainstorm creative marketing campaigns",
                "generate ideas for community art project",
                "develop unique approach to employee engagement",
                "create innovative educational activities",
                "brainstorm creative ways to reduce plastic waste",
                "generate ideas for interactive museum exhibit",
                
                # Music & Performance
                "compose melody for love song",
                "create lyrics about social justice",
                "develop choreography for dance performance",
                "write script for short film",
                "compose music for video game soundtrack"
            ],
            'BUSINESS_PROFESSIONAL': [
                # Business Strategy & Planning
                "develop comprehensive business plan for new startup",
                "create marketing strategy for product launch",
                "analyze competitive landscape and market positioning",
                "develop pricing strategy for subscription service",
                "create financial projections for business expansion",
                "design organizational structure for growing company",
                "develop customer acquisition strategy",
                "create employee retention program",
                "analyze market trends and opportunities",
                "develop partnership strategy for business growth",
                
                # Professional Communication
                "write professional email to potential client",
                "create presentation for board meeting",
                "draft proposal for new project initiative",
                "write job description for software engineer position",
                "create performance review feedback for employee",
                "draft contract terms for service agreement",
                "write professional recommendation letter",
                "create agenda for quarterly team meeting",
                "draft press release for product announcement",
                "write executive summary for business report",
                
                # Finance & Operations
                "prepare financial forecast presentation",
                "analyze budget allocation for marketing department",
                "create investment proposal for venture capital",
                "develop cost-benefit analysis for new technology",
                "prepare quarterly earnings report",
                "analyze return on investment for projects",
                "create cash flow projections",
                "develop risk assessment for business decisions",
                
                # Legal & Compliance
                "understand legal requirements for business registration",
                "draft terms of service for website",
                "create privacy policy for mobile app",
                "understand intellectual property protection",
                "draft non-disclosure agreement for partnership",
                "understand employment law compliance",
                "create data protection policy",
                "understand contract law principles",
                "draft licensing agreement for software",
                "understand regulatory compliance requirements",
                
                # Management & Leadership
                "develop leadership training program",
                "create conflict resolution procedures",
                "design performance management system",
                "develop succession planning strategy",
                "create change management process"
            ],
            'CONVERSATIONAL_ADVICE': [
                # Personal Life Advice
                "what should I do about relationship problems with partner",
                "how to deal with stress at work effectively",
                "give me advice on career change decisions",
                "help me choose between two job offers",
                "what are good strategies for work-life balance",
                "how to improve communication with family members",
                "advice on managing personal finances better",
                "help me decide on living situation",
                "what should I consider when buying first car",
                "advice on maintaining healthy relationships",
                
                # Lifestyle & Health Recommendations
                "recommend good books for summer reading",
                "suggest healthy meal ideas for busy schedule",
                "what are good exercises for beginners",
                "recommend travel destinations for families",
                "suggest hobbies for creative expression",
                "what are good ways to learn new skills",
                "recommend strategies for better sleep",
                "suggest ways to stay motivated during difficult times",
                "what are good practices for mental health",
                "recommend ways to build social connections",
                
                # Opinion & Discussion
                "what's your opinion on remote work trends",
                "discuss pros and cons of social media",
                "what do you think about current technology trends",
                "share thoughts on environmental conservation",
                "discuss benefits of lifelong learning",
                "what are your views on work-life integration",
                "discuss importance of cultural diversity",
                "share opinion on digital privacy concerns",
                "discuss impact of artificial intelligence on society",
                "what are your thoughts on sustainable living practices",
                
                # Decision Support
                "help me weigh pros and cons of moving to new city",
                "advice on whether to pursue graduate degree",
                "should I start my own business or stay employed",
                "help me decide on vacation destination",
                "advice on choosing between different career paths",
                "what factors should I consider when changing jobs",
                "help me decide on major life changes",
                "advice on balancing personal and professional goals"
            ]
        }
        
        # Enhanced embedding computation with augmentation techniques
        try:
            
            for category, examples in category_examples.items():
                
                # Basic embedding computation
                embeddings = []
                
                # Encode original examples
                for example in examples:
                    embedding = self.embedding_model.encode(example)
                    embeddings.append(embedding)
                
                # Add paraphrase augmentation if NLTK is available
                if NLTK_AVAILABLE:
                    augmented_examples = self._generate_paraphrases(examples[:10])  # Limit for performance
                    
                    for aug_example in augmented_examples:
                        embedding = self.embedding_model.encode(aug_example)
                        embeddings.append(embedding)
                
                # Compute category representation using centroid
                if embeddings:
                    category_embedding = np.mean(embeddings, axis=0)
                    self.category_embeddings[category] = category_embedding
                else:
                    logger.warning(f"No embeddings computed for category '{category}'")
                    
        except Exception as e:
            logger.warning(f"Enhanced embedding computation failed: {e}")
            
            # Fallback to basic computation
            for category, examples in category_examples.items():
                try:
                    embeddings = [self.embedding_model.encode(example) for example in examples]
                    category_embedding = np.mean(embeddings, axis=0)
                    self.category_embeddings[category] = category_embedding
                except Exception as fallback_error:
                    logger.error(f"Failed to compute embedding for '{category}': {fallback_error}")
    
    def _generate_paraphrases(self, examples: List[str]) -> List[str]:
        """Generate paraphrased versions of examples for data augmentation"""
        if not NLTK_AVAILABLE:
            return []
        
        paraphrases = []
        
        try:
            from nltk.corpus import wordnet
            
            for example in examples:
                # Simple synonym-based paraphrasing
                tokens = word_tokenize(example.lower())
                paraphrased_tokens = []
                
                for token in tokens:
                    # Get synonyms from WordNet
                    synonyms = set()
                    for syn in wordnet.synsets(token):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym != token and len(synonym) > 2:
                                synonyms.add(synonym)
                    
                    # Use first synonym if available, otherwise keep original
                    if synonyms and len(synonyms) > 0:
                        paraphrased_tokens.append(list(synonyms)[0])
                    else:
                        paraphrased_tokens.append(token)
                
                paraphrase = ' '.join(paraphrased_tokens)
                if paraphrase != example.lower() and len(paraphrase) > 10:
                    paraphrases.append(paraphrase)
                    
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
        
        return paraphrases[:5]  # Limit to 5 paraphrases per original

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
        """Enhanced rule-based classification with advanced NLP features and scoring"""
        import re
        
        logger.info("Starting enhanced rule-based classification")
        query_text = processed_query.normalized_query.lower()
        original_text = processed_query.original_query.lower()
        
        # Initialize lexicon builder for domain term analysis
        if not hasattr(self, 'lexicon_builder'):
            self.lexicon_builder = DomainLexiconBuilder()
        
        # Enhanced intent pattern recognition
        intent_patterns = {
            'how_to': r'\b(how\s+to|how\s+do\s+i|how\s+can\s+i)\b',
            'what_is': r'\b(what\s+is|what\s+are|what\s+does)\b',
            'why': r'\b(why\s+is|why\s+do|why\s+does|why\s+are)\b',
            'explain': r'\b(explain|describe|tell\s+me\s+about)\b',
            'create': r'\b(create|build|make|generate|write)\b',
            'fix': r'\b(fix|debug|solve|resolve|troubleshoot)\b',
            'compare': r'\b(compare|difference|versus|vs)\b',
            'analyze': r'\b(analyze|examine|study|investigate)\b',
            'optimize': r'\b(optimize|improve|enhance|maximize|minimize)\b',
            'find': r'\b(find|locate|identify|detect|discover)\b',
            'calculate': r'\b(calculate|compute|measure|determine)\b'
        }
        
        # Detect intent patterns
        detected_intents = []
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query_text):
                detected_intents.append(intent)
        
        # Question type detection
        question_type = 'general'
        question_patterns = {
            'factual': r'^(what|when|where|who|which)\b',
            'procedural': r'^(how|how\s+to|how\s+do|how\s+can)\b',
            'causal': r'^(why|what\s+causes|what\s+makes)\b',
            'comparative': r'\b(better|worse|difference|compare|versus|vs)\b',
            'evaluative': r'\b(should|would|could|recommend|suggest|opinion)\b',
            'creative': r'\b(create|write|design|compose|generate)\b',
            'analytical': r'\b(analyze|examine|study|investigate|research)\b'
        }
        
        for q_type, pattern in question_patterns.items():
            if re.search(pattern, query_text):
                question_type = q_type
                break
        
        # Domain density analysis
        tokens = query_text.split()
        content_tokens = [token for token in tokens if len(token) > 2]  # Filter short words
        total_tokens = len(content_tokens) if content_tokens else 1
        
        domain_densities = {}
        for domain in ['technical', 'business', 'academic']:
            domain_terms = self.lexicon_builder.get_domain_terms(domain)
            found_terms = [token for token in content_tokens if token in domain_terms]
            domain_densities[domain] = len(found_terms) / total_tokens
        
        # Enhanced category scoring with comprehensive rules
        enhanced_patterns = {
            'CODE_TECHNICAL': {
                'required_features': {
                    'domain_density': ('technical', 0.2),  # At least 20% technical terms
                    'intent_patterns': ['create', 'fix', 'how_to', 'optimize', 'find'],
                    'question_types': ['procedural', 'creative', 'analytical']
                },
                'bonus_features': {
                    'high_tech_density': ('technical', 0.4),  # Bonus for very high tech density
                    'specific_patterns': [r'\b(iou|intersection|union)\b', r'\b(cell|tissue|distance)\b'],
                    'multiple_intents': 2  # Bonus for multiple matching intents
                },
                'negative_indicators': {
                    'domain_density': [('business', 0.3), ('academic', 0.4)]  # Penalties for other domains
                }
            },
            'MATHEMATICAL_SCIENTIFIC': {
                'required_features': {
                    'domain_density': ('academic', 0.15),
                    'intent_patterns': ['analyze', 'explain', 'what_is', 'calculate'],
                    'question_types': ['factual', 'analytical', 'causal']
                },
                'bonus_features': {
                    'research_patterns': [r'\b(benefits.*of.*renewable|economic.*impact|research.*on)\b'],
                    'scientific_terms': [r'\b(correlation|hypothesis|methodology)\b']
                },
                'negative_indicators': {
                    'domain_density': [('business', 0.2)]
                }
            },
            'EDUCATIONAL_ACADEMIC': {
                'required_features': {
                    'domain_density': ('academic', 0.1),
                    'intent_patterns': ['explain', 'what_is', 'why', 'how_to'],
                    'question_types': ['factual', 'procedural', 'causal']
                },
                'bonus_features': {
                    'educational_patterns': [r'\b(benefits.*of|advantages.*of|what.*are)\b'],
                    'learning_context': [r'\b(teach|learn|study|understand)\b']
                },
                'negative_indicators': {}
            },
            'CREATIVE_ARTISTIC': {
                'required_features': {
                    'intent_patterns': ['create'],
                    'question_types': ['creative', 'procedural']
                },
                'bonus_features': {
                    'creative_patterns': [r'\b(write.*story|create.*poem|design.*logo)\b'],
                    'artistic_terms': [r'\b(narrative|character|plot|artistic|compose)\b']
                },
                'negative_indicators': {
                    'domain_density': [('technical', 0.3), ('business', 0.3)]
                }
            },
            'BUSINESS_PROFESSIONAL': {
                'required_features': {
                    'domain_density': ('business', 0.15),
                    'intent_patterns': ['create', 'analyze', 'compare', 'explain'],
                    'question_types': ['procedural', 'analytical', 'evaluative']
                },
                'bonus_features': {
                    'legal_patterns': [r'\b(lawsuit|legal|contract|compliance)\b'],
                    'business_context': [r'\b(professional|strategy|marketing)\b']
                },
                'negative_indicators': {}
            },
            'CONVERSATIONAL_ADVICE': {
                'required_features': {
                    'intent_patterns': ['how_to', 'what_is', 'why'],
                    'question_types': ['evaluative', 'procedural', 'general']
                },
                'bonus_features': {
                    'advice_patterns': [r'\b(advice|help.*me|what.*should.*i|recommend)\b'],
                    'personal_context': [r'\b(personal|relationship|lifestyle)\b']
                },
                'negative_indicators': {
                    'domain_density': [('technical', 0.2), ('business', 0.2), ('academic', 0.2)]
                }
            }
        }
        
        category_scores = {}
        
        for category, rules in enhanced_patterns.items():
            total_score = 0.0
            matches_found = []
            
            # Required features scoring
            required = rules.get('required_features', {})
            
            # Domain density requirements
            if 'domain_density' in required:
                domain, threshold = required['domain_density']
                actual_density = domain_densities.get(domain, 0)
                if actual_density >= threshold:
                    total_score += 0.4  # Base score for meeting requirement
                    total_score += actual_density * 0.3  # Bonus for exceeding
                    matches_found.append(f"DOMAIN_DENSITY: {domain} ({actual_density:.3f})")
            
            # Intent pattern requirements
            if 'intent_patterns' in required:
                required_intents = set(required['intent_patterns'])
                found_intents = set(detected_intents)
                matching_intents = required_intents.intersection(found_intents)
                if matching_intents:
                    total_score += 0.3
                    total_score += len(matching_intents) * 0.1  # Bonus for multiple matches
                    matches_found.append(f"INTENT: {matching_intents}")
            
            # Question type requirements
            if 'question_types' in required:
                required_types = set(required['question_types'])
                if question_type in required_types:
                    total_score += 0.2
                    matches_found.append(f"QUESTION_TYPE: {question_type}")
            
            # Bonus features
            bonus = rules.get('bonus_features', {})
            
            # High domain density bonus
            if 'high_tech_density' in bonus:
                domain, threshold = bonus['high_tech_density']
                if domain_densities.get(domain, 0) >= threshold:
                    total_score += 0.2
                    matches_found.append(f"HIGH_DENSITY_BONUS: {domain}")
            
            # Specific pattern bonuses
            if 'specific_patterns' in bonus:
                for pattern in bonus['specific_patterns']:
                    if re.search(pattern, query_text):
                        total_score += 0.15
                        matches_found.append(f"SPECIFIC_PATTERN: {pattern}")
            
            # Research pattern bonuses
            if 'research_patterns' in bonus:
                for pattern in bonus['research_patterns']:
                    if re.search(pattern, query_text):
                        total_score += 0.15
                        matches_found.append(f"RESEARCH_PATTERN: {pattern}")
            
            # Other pattern bonuses
            for bonus_type in ['educational_patterns', 'creative_patterns', 'legal_patterns', 
                              'business_context', 'advice_patterns', 'personal_context', 
                              'scientific_terms', 'artistic_terms', 'learning_context']:
                if bonus_type in bonus:
                    for pattern in bonus[bonus_type]:
                        if re.search(pattern, query_text):
                            total_score += 0.1
                            matches_found.append(f"{bonus_type.upper()}: {pattern}")
            
            # Multiple intents bonus
            if 'multiple_intents' in bonus:
                threshold = bonus['multiple_intents']
                if len(detected_intents) >= threshold:
                    total_score += 0.15
                    matches_found.append(f"MULTIPLE_INTENTS: {len(detected_intents)}")
            
            # Negative indicators (penalties)
            negative = rules.get('negative_indicators', {})
            
            if 'domain_density' in negative:
                for domain, threshold in negative['domain_density']:
                    actual_density = domain_densities.get(domain, 0)
                    if actual_density > threshold:
                        penalty = (actual_density - threshold) * 0.4
                        total_score -= penalty
                        matches_found.append(f"PENALTY: {domain} density ({actual_density:.3f})")
            
            # Ensure score is non-negative
            total_score = max(0.0, total_score)
            
            category_scores[category] = {
                'score': total_score,
                'matches': matches_found
            }
            
            if matches_found:
                logger.info(f"Category '{category}': {total_score:.4f} - {matches_found}")
        
        # Find best category
        if not category_scores or all(score['score'] == 0 for score in category_scores.values()):
            logger.warning("No enhanced rule matches found, using fallback")
            return 'CONVERSATIONAL_ADVICE', 0.1
        
        best_category = max(category_scores, key=lambda x: category_scores[x]['score'])
        best_score = category_scores[best_category]['score']
        
        # Enhanced confidence calculation
        confidence = min(best_score, 1.0)  # Cap at 1.0
        
        # Bonus for very strong matches
        if best_score >= 0.8:
            confidence = min(confidence + 0.1, 1.0)
            logger.info("Strong match confidence boost applied")
        
        logger.info(f"Enhanced rule-based result: '{best_category}' with confidence {confidence:.4f}")
        logger.info(f"Domain densities: {domain_densities}")
        logger.info(f"Detected intents: {detected_intents}")
        logger.info(f"Question type: {question_type}")
        
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
    # test_query = "If I want to file a lawsuit against a company, what steps should I take?"
    queries = ["You're a senior software engineer. How do I debug this React component in the following code?",
        "We have this article about DDPM. We want to differentiate this method fro mstable diffusion and DDIM, be concise",
        "We're contemplating using LDA or NMF for topic modeling. Which one is better for our dataset?",]
    for test_query in queries:
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