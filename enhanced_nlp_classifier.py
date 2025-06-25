"""
Enhanced Query Classification System with Advanced NLP Techniques
Author: Senior Data Scientist (10+ years NLP experience)
"""

import nltk
import logging
import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class EnhancedQueryFeatures:
    """Enhanced feature set for query classification"""
    # Linguistic features
    tokens: List[str] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)
    named_entities: List[str] = field(default_factory=list)
    
    # Domain-specific features
    technical_terms: List[str] = field(default_factory=list)
    business_terms: List[str] = field(default_factory=list)
    academic_terms: List[str] = field(default_factory=list)
    creative_terms: List[str] = field(default_factory=list)
    scientific_terms: List[str] = field(default_factory=list)
    
    # Semantic features
    intent_patterns: List[str] = field(default_factory=list)
    question_type: Optional[str] = None
    complexity_indicators: Dict[str, float] = field(default_factory=dict)
    conversational_framing: List[str] = field(default_factory=list)  # New field
    
    # Statistical features
    domain_densities: Dict[str, float] = field(default_factory=dict)
    lexical_diversity: float = 0.0
    semantic_coherence: float = 0.0

class DomainLexiconBuilder:
    """Builds comprehensive domain-specific lexicons using NLTK and external resources"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Base domain lexicons - these will be expanded
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
                    'distribution', 'hypothesis', 'inference', 'significance',
                    # Computer vision and image processing
                    'computer', 'vision', 'image', 'processing', 'segmentation', 'detection',
                    'iou', 'intersection', 'union', 'bounding', 'box', 'mask', 'pixel',
                    'opencv', 'convolution', 'filter', 'edge', 'contour', 'morphology',
                    # Biomedical and scientific computing
                    'biomedical', 'microscopy', 'cell', 'tissue', 'medical', 'imaging',
                    'analysis', 'measurement', 'quantification', 'visualization'
                ],
                'web_development': [
                    'frontend', 'backend', 'server', 'client', 'database', 'query',
                    'endpoint', 'authentication', 'authorization', 'session', 'cookie',
                    'middleware', 'routing', 'deployment', 'scaling', 'optimization'
                ],
                'systems': [
                    'operating', 'memory', 'processor', 'thread', 'process', 'kernel',
                    'filesystem', 'network', 'protocol', 'security', 'encryption',
                    'performance', 'benchmark', 'monitoring', 'logging'
                ]
            },
            'business': {
                'finance': [
                    'revenue', 'profit', 'budget', 'investment', 'return', 'capital',
                    'expense', 'income', 'cash', 'flow', 'balance', 'sheet', 'statement',
                    'accounting', 'audit', 'tax', 'compliance', 'regulation'
                ],
                'marketing': [
                    'campaign', 'brand', 'customer', 'segment', 'target', 'audience',
                    'conversion', 'engagement', 'retention', 'acquisition', 'funnel',
                    'analytics', 'metrics', 'performance', 'roi', 'ctr', 'impression'
                ],
                'management': [
                    'strategy', 'planning', 'execution', 'leadership', 'team', 'project',
                    'deadline', 'milestone', 'stakeholder', 'communication', 'delegation',
                    'motivation', 'performance', 'evaluation', 'feedback', 'development'
                ],
                'legal': [
                    'contract', 'agreement', 'clause', 'liability', 'negligence',
                    'compliance', 'regulation', 'statute', 'lawsuit', 'litigation',
                    'intellectual', 'property', 'trademark', 'copyright', 'patent'
                ]
            },
            'academic': {
                'research': [
                    'hypothesis', 'methodology', 'experiment', 'observation', 'analysis',
                    'conclusion', 'literature', 'review', 'citation', 'reference',
                    'peer', 'journal', 'publication', 'conference', 'abstract'
                ],
                'education': [
                    'curriculum', 'pedagogy', 'assessment', 'evaluation', 'learning',
                    'teaching', 'instruction', 'student', 'teacher', 'professor',
                    'lecture', 'seminar', 'assignment', 'thesis', 'dissertation'
                ],
                'science': [
                    'theory', 'principle', 'law', 'equation', 'formula', 'calculation',
                    'measurement', 'experiment', 'laboratory', 'data', 'result',
                    'evidence', 'proof', 'validation', 'verification'
                ]
            },
            'creative': {
                'writing': [
                    'narrative', 'character', 'plot', 'theme', 'setting', 'dialogue',
                    'style', 'tone', 'voice', 'perspective', 'structure', 'genre',
                    'fiction', 'poetry', 'prose', 'metaphor', 'symbolism'
                ],
                'design': [
                    'aesthetic', 'composition', 'color', 'typography', 'layout',
                    'visual', 'graphic', 'illustration', 'branding', 'identity',
                    'user', 'interface', 'experience', 'wireframe', 'prototype'
                ],
                'arts': [
                    'painting', 'sculpture', 'music', 'dance', 'theater', 'film',
                    'photography', 'drawing', 'sketch', 'performance', 'exhibition',
                    'gallery', 'museum', 'artist', 'creativity', 'inspiration'
                ]
            }
        }
        
        self.expanded_lexicons = self._expand_lexicons()
    
    def _expand_lexicons(self) -> Dict[str, Set[str]]:
        """Expand base lexicons using WordNet synsets and morphological variations"""
        logger.info("Expanding domain lexicons using WordNet and morphological analysis")
        
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
                    
                    # Add WordNet synsets
                    synsets = wordnet.synsets(term) # synsets is a list of synonyms
                    for synset in synsets[:3]:  # Limit to top 3 synsets
                        for lemma in synset.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if len(synonym) > 2:  # Filter out very short synonyms
                                expanded[key].add(synonym)
                    
                    # Add stem and lemma
                    expanded[key].add(self.stemmer.stem(term))
                    expanded[key].add(self.lemmatizer.lemmatize(term))
        
        # Create aggregate domain lexicons
        for domain in ['technical', 'business', 'academic', 'creative']:
            expanded[domain] = set()
            for key in expanded.keys():
                if key.startswith(domain + '_'):
                    expanded[domain].update(expanded[key])
        
        logger.info(f"Expanded lexicons created for {len(expanded)} domains")
        for domain, terms in expanded.items():
            logger.info(f"Domain '{domain}': {len(terms)} terms")
        
        return dict(expanded)
    
    def get_domain_terms(self, domain: str) -> Set[str]:
        """Get all terms for a specific domain"""
        return self.expanded_lexicons.get(domain, set())
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains"""
        return list(self.expanded_lexicons.keys())

class EnhancedNLPPreprocessor:
    """Advanced NLP preprocessing with linguistic analysis"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lexicon_builder = DomainLexiconBuilder()
        
        # Intent patterns using regex
        self.intent_patterns = {
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
        
        # Conversational markers that indicate user is framing a technical question conversationally
        self.conversational_framing_patterns = {
            'role_based': r'\b(you\'re\s+a|as\s+a|being\s+a)\s+(senior|experienced|expert|professional)\s+(engineer|developer|scientist|researcher|analyst)\b',
            'collaborative': r'\b(we\s+have|we\s+want|we\'re\s+contemplating|we\s+need|we\s+are)\b',
            'contextual': r'\b(in\s+the\s+following|with\s+respect\s+to|regarding|concerning)\b',
            'professional': r'\b(our\s+dataset|our\s+system|our\s+project|our\s+team)\b'
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'technical_jargon': r'\b(implementation|architecture|optimization|scalability|performance)\b',
            'academic_language': r'\b(hypothesis|methodology|empirical|theoretical|systematic)\b',
            'formal_language': r'\b(furthermore|consequently|nevertheless|accordingly|henceforth)\b',
            'code_elements': r'[{}()\[\];]|def\s+|class\s+|import\s+|function\s*\(',
            'mathematical': r'[∑∏∫∂∇]|\\sum|\\int|\\frac|equation|formula|theorem'
        }
    
    def extract_enhanced_features(self, query: str) -> EnhancedQueryFeatures:
        """Extract comprehensive linguistic and semantic features"""
        logger.info(f"Extracting enhanced features for query: '{query[:100]}...'")
        
        features = EnhancedQueryFeatures()
        
        # Basic tokenization
        features.tokens = word_tokenize(query.lower())
        features.tokens = [token for token in features.tokens if token.isalnum()]
        
        # Remove stopwords for analysis
        content_tokens = [token for token in features.tokens if token not in self.stop_words]
        
        # Stemming and lemmatization
        features.stems = [self.stemmer.stem(token) for token in content_tokens]
        features.lemmas = [self.lemmatizer.lemmatize(token) for token in content_tokens]
        
        # POS tagging
        features.pos_tags = pos_tag(features.tokens)
        
        # Named entity recognition
        try:
            chunks = ne_chunk(features.pos_tags)
            features.named_entities = [
                ' '.join([token for token, pos in chunk.leaves()])
                for chunk in chunks if hasattr(chunk, 'label')
            ]
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            features.named_entities = []
        
        # Domain-specific term extraction
        query_lower = query.lower()
        all_terms = features.tokens + features.stems + features.lemmas
        
        for domain in ['technical', 'business', 'academic', 'creative']:
            domain_terms = self.lexicon_builder.get_domain_terms(domain)
            found_terms = [term for term in all_terms if term in domain_terms]
            
            # Also check for multi-word terms in original query
            for term in domain_terms:
                if ' ' in term and term in query_lower:
                    found_terms.append(term)
            
            if domain == 'technical':
                features.technical_terms = list(set(found_terms))
            elif domain == 'business':
                features.business_terms = list(set(found_terms))
            elif domain == 'academic':
                features.academic_terms = list(set(found_terms))
            elif domain == 'creative':
                features.creative_terms = list(set(found_terms))
        
        # Scientific terms (subset of academic)
        scientific_domains = ['technical_data_science', 'academic_science']
        features.scientific_terms = []
        for domain in scientific_domains:
            domain_terms = self.lexicon_builder.get_domain_terms(domain)
            features.scientific_terms.extend([term for term in all_terms if term in domain_terms])
        features.scientific_terms = list(set(features.scientific_terms))
        
        # Intent pattern matching
        features.intent_patterns = []
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                features.intent_patterns.append(intent)
        
        # Detect conversational framing (indicates technical query phrased conversationally)
        features.conversational_framing = []
        for framing_type, pattern in self.conversational_framing_patterns.items():
            if re.search(pattern, query_lower):
                features.conversational_framing.append(framing_type)
        
        # Question type detection
        features.question_type = self._detect_question_type(query)
        
        # Complexity analysis
        features.complexity_indicators = {}
        for indicator, pattern in self.complexity_indicators.items():
            matches = len(re.findall(pattern, query_lower))
            features.complexity_indicators[indicator] = matches / len(features.tokens) if features.tokens else 0
        
        # Domain density calculation
        total_tokens = len(content_tokens) if content_tokens else 1
        features.domain_densities = {
            'technical': len(features.technical_terms) / total_tokens,
            'business': len(features.business_terms) / total_tokens,
            'academic': len(features.academic_terms) / total_tokens,
            'creative': len(features.creative_terms) / total_tokens,
            'scientific': len(features.scientific_terms) / total_tokens
        }
        
        # Lexical diversity (Type-Token Ratio)
        features.lexical_diversity = len(set(content_tokens)) / len(content_tokens) if content_tokens else 0
        
        logger.info(f"Enhanced feature extraction completed")
        logger.info(f"Domain densities: {features.domain_densities}")
        logger.info(f"Intent patterns: {features.intent_patterns}")
        logger.info(f"Conversational framing: {features.conversational_framing}")
        
        return features
    
    def _detect_question_type(self, query: str) -> Optional[str]:
        """Detect the type of question being asked"""
        query_lower = query.lower().strip()
        
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
            if re.search(pattern, query_lower):
                return q_type
        
        return 'general'

class EnhancedRuleBasedClassifier:
    """Advanced rule-based classifier using comprehensive NLP features"""
    
    def __init__(self):
        self.preprocessor = EnhancedNLPPreprocessor()
        
        # Enhanced classification rules with weights
        self.classification_rules = {
            'CODE_TECHNICAL': {
                'required_features': {
                    'technical_terms': 0.15,  # Reduced from 0.3 - allow conversational technical queries
                    'intent_patterns': ['create', 'fix', 'how_to', 'optimize', 'find', 'calculate'],
                    'question_types': ['procedural', 'creative', 'analytical', 'comparative']  # Added comparative
                },
                'bonus_features': {
                    'complexity_indicators': ['code_elements', 'technical_jargon'],
                    'pos_patterns': ['VB', 'NN'],  # Verbs and nouns
                    'named_entities': True,
                    'scientific_terms': 0.1,  # Scientific + technical combination
                    'conversational_framing': 0.3  # Bonus for conversational framing of technical questions
                },
                'negative_indicators': {
                    'creative_terms': 0.3,  # Too many creative terms reduce score
                    'business_terms': 0.4   # Too many business terms reduce score
                }
            },
            'MATHEMATICAL_SCIENTIFIC': {
                'required_features': {
                    'scientific_terms': 0.1,  # Reduced from 0.2
                    'academic_terms': 0.1,    # Reduced from 0.2
                    'intent_patterns': ['analyze', 'explain', 'what_is', 'compare', 'calculate'],
                    'question_types': ['factual', 'analytical', 'causal', 'comparative']
                },
                'bonus_features': {
                    'complexity_indicators': ['mathematical', 'academic_language'],
                    'technical_terms': 0.15,  # Increased - scientific often overlaps with technical
                    'conversational_framing': 0.2  # Bonus for conversational framing
                },
                'negative_indicators': {
                    'creative_terms': 0.4,
                    'business_terms': 0.3
                }
            },
            'EDUCATIONAL_ACADEMIC': {
                'required_features': {
                    'academic_terms': 0.1,  # Reduced from 0.2
                    'intent_patterns': ['explain', 'what_is', 'why', 'how_to'],
                    'question_types': ['factual', 'procedural', 'causal']
                },
                'bonus_features': {
                    'complexity_indicators': ['academic_language', 'formal_language'],
                    'scientific_terms': 0.15,
                    'technical_terms': 0.1,  # Allow some technical terms in education
                    'lexical_diversity': 0.5,  # Educational content tends to be diverse
                    'conversational_framing': 0.15  # Smaller bonus for conversational framing
                },
                'negative_indicators': {
                    'business_terms': 0.3,
                    'creative_terms': 0.2
                }
            },
            'CREATIVE_ARTISTIC': {
                'required_features': {
                    'creative_terms': 0.2,
                    'intent_patterns': ['create', 'how_to'],
                    'question_types': ['creative', 'procedural']
                },
                'bonus_features': {
                    'pos_patterns': ['JJ', 'RB'],  # Adjectives and adverbs
                    'lexical_diversity': 0.6
                },
                'negative_indicators': {
                    'technical_terms': 0.4,  # Increased penalty
                    'business_terms': 0.4,   # Increased penalty
                    'scientific_terms': 0.4, # Increased penalty
                    'conversational_framing': 0.1  # Slight penalty for professional framing
                }
            },
            'BUSINESS_PROFESSIONAL': {
                'required_features': {
                    'business_terms': 0.15,  # Reduced from 0.2
                    'intent_patterns': ['create', 'analyze', 'compare', 'explain'],
                    'question_types': ['procedural', 'analytical', 'evaluative', 'comparative']
                },
                'bonus_features': {
                    'complexity_indicators': ['formal_language'],
                    'named_entities': True,
                    'conversational_framing': 0.25  # Bonus for professional conversational framing
                },
                'negative_indicators': {
                    'creative_terms': 0.4,
                    'technical_terms': 0.05  # Reduced penalty - business often involves technical discussions
                }
            },
            'CONVERSATIONAL_ADVICE': {
                'required_features': {
                    'intent_patterns': ['how_to', 'what_is', 'why'],
                    'question_types': ['evaluative', 'procedural', 'general']
                },
                'bonus_features': {
                    'lexical_diversity': 0.3  # Conversational tends to be simpler
                },
                'negative_indicators': {
                    'technical_terms': 0.15,      # Reduced penalty
                    'business_terms': 0.15,       # Reduced penalty
                    'academic_terms': 0.15,       # Reduced penalty
                    'scientific_terms': 0.15,     # Reduced penalty
                    'conversational_framing': 0.4  # Strong penalty for professional framing
                }
            }
        }
    
    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query using enhanced rule-based approach"""
        logger.info(f"Starting enhanced rule-based classification for: '{query[:100]}...'")
        
        # Extract features
        features = self.preprocessor.extract_enhanced_features(query)
        
        # Calculate scores for each category
        category_scores = {}
        
        for category, rules in self.classification_rules.items():
            score = self._calculate_category_score(features, rules)
            category_scores[category] = score
            logger.info(f"Category '{category}' score: {score:.4f}")
        
        # Find best category
        if not category_scores or max(category_scores.values()) == 0:
            return 'CONVERSATIONAL_ADVICE', 0.1
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        # Normalize confidence to [0, 1] range
        max_possible_score = 1.0  # Theoretical maximum
        confidence = min(confidence, max_possible_score)
        
        logger.info(f"Enhanced rule-based classification result: '{best_category}' with confidence {confidence:.4f}")
        
        return best_category, confidence
    
    def _calculate_category_score(self, features: EnhancedQueryFeatures, rules: Dict) -> float:
        """Calculate score for a specific category based on rules"""
        score = 0.0
        
        # Required features
        required = rules.get('required_features', {})
        
        # Domain density requirements
        for domain in ['technical_terms', 'business_terms', 'academic_terms', 'creative_terms', 'scientific_terms']:
            if domain in required:
                required_density = required[domain]
                actual_density = features.domain_densities.get(domain.replace('_terms', ''), 0)
                if actual_density >= required_density:
                    score += 0.3  # Base score for meeting requirement
                    score += actual_density * 0.2  # Bonus for exceeding requirement
        
        # Intent pattern requirements
        if 'intent_patterns' in required:
            required_intents = set(required['intent_patterns'])
            found_intents = set(features.intent_patterns)
            if required_intents.intersection(found_intents):
                score += 0.2
                # Bonus for multiple matching intents
                score += len(required_intents.intersection(found_intents)) * 0.1
        
        # Question type requirements
        if 'question_types' in required:
            required_types = set(required['question_types'])
            if features.question_type in required_types:
                score += 0.2
        
        # Bonus features
        bonus = rules.get('bonus_features', {})
        
        # Complexity indicators bonus
        if 'complexity_indicators' in bonus:
            bonus_indicators = bonus['complexity_indicators']
            for indicator in bonus_indicators:
                if indicator in features.complexity_indicators:
                    score += features.complexity_indicators[indicator] * 0.1
        
        # Lexical diversity bonus
        if 'lexical_diversity' in bonus:
            threshold = bonus['lexical_diversity']
            if features.lexical_diversity >= threshold:
                score += 0.1
        
        # Named entities bonus
        if bonus.get('named_entities') and features.named_entities:
            score += 0.1
        
        # POS pattern bonus
        if 'pos_patterns' in bonus:
            pos_tags = [tag for word, tag in features.pos_tags]
            required_pos = bonus['pos_patterns']
            pos_matches = sum(1 for tag in pos_tags if tag in required_pos)
            if pos_matches > 0:
                score += (pos_matches / len(pos_tags)) * 0.1 if pos_tags else 0
        
        # Conversational framing bonus
        if 'conversational_framing' in bonus and features.conversational_framing:
            framing_score = len(features.conversational_framing) * bonus['conversational_framing']
            score += framing_score
        
        # Negative indicators (penalties)
        negative = rules.get('negative_indicators', {})
        
        for domain, threshold in negative.items():
            if domain.endswith('_terms'):
                domain_key = domain.replace('_terms', '')
                actual_density = features.domain_densities.get(domain_key, 0)
                if actual_density > threshold:
                    penalty = (actual_density - threshold) * 0.3
                    score -= penalty
            elif domain == 'conversational_framing':
                # Apply penalty if conversational framing is present
                if features.conversational_framing:
                    penalty = len(features.conversational_framing) * threshold
                    score -= penalty
        
        # Ensure score is non-negative
        score = max(0.0, score)
        
        return score

class EnhancedEmbeddingClassifier:
    """Enhanced embedding-based classifier with domain-specific training data"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.category_embeddings = {}
        self.synthetic_data_generator = SyntheticQueryGenerator()
        
        # Initialize with enhanced category representations
        self._initialize_enhanced_embeddings()
    
    def _initialize_enhanced_embeddings(self):
        """Initialize embeddings with comprehensive category representations"""
        logger.info("Initializing enhanced category embeddings")
        
        # Generate comprehensive training examples for each category
        category_data = {
            'CODE_TECHNICAL': self._generate_technical_examples(),
            'MATHEMATICAL_SCIENTIFIC': self._generate_scientific_examples(),
            'EDUCATIONAL_ACADEMIC': self._generate_educational_examples(),
            'CREATIVE_ARTISTIC': self._generate_creative_examples(),
            'BUSINESS_PROFESSIONAL': self._generate_business_examples(),
            'CONVERSATIONAL_ADVICE': self._generate_conversational_examples()
        }
        
        # Compute embeddings for each category
        for category, examples in category_data.items():
            logger.info(f"Computing embeddings for {category} using {len(examples)} examples")
            
            # Encode all examples
            embeddings = self.embedding_model.encode(examples)
            
            # Use centroid as category representation
            category_embedding = np.mean(embeddings, axis=0)
            self.category_embeddings[category] = category_embedding
            
            logger.info(f"Category {category} embedding computed, shape: {category_embedding.shape}")
    
    def _generate_technical_examples(self) -> List[str]:
        """Generate comprehensive technical query examples"""
        return [
            # Programming queries - now with conversational framing
            "write a python function to sort a list of numbers",
            "debug this javascript error in my React component",
            "how to implement binary search algorithm in C++",
            "optimize SQL query performance for large datasets",
            "fix compilation error in my Java program",
            "create REST API endpoint using FastAPI",
            "implement user authentication in Node.js",
            "design database schema for e-commerce application",
            "troubleshoot memory leak in Python application",
            "configure Docker container for microservice deployment",
            
            # Conversational technical queries - added to improve classification
            "You're a senior software engineer. How do I debug this React component?",
            "As an experienced developer, what's the best way to optimize this SQL query?",
            "We have a performance issue with our microservice. How to troubleshoot?",
            "I'm a junior developer. Can you help me implement authentication?",
            "Our team is struggling with this deployment issue. Any suggestions?",
            "Being a DevOps engineer, how would you configure this Docker setup?",
            "We're contemplating migrating to microservices. What's your approach?",
            "As a technical lead, how do you handle code reviews efficiently?",
            "Our startup needs to implement CI/CD. What tools do you recommend?",
            "We have legacy code that needs refactoring. Best practices?",
            
            # Data science queries - now with conversational framing
            "build machine learning model for image classification",
            "implement neural network using TensorFlow",
            "optimize hyperparameters for deep learning model",
            "create data pipeline for ETL processing",
            "implement recommendation system using collaborative filtering",
            "analyze time series data using LSTM networks",
            "build computer vision model for object detection",
            "implement natural language processing for sentiment analysis",
            "create automated feature engineering pipeline",
            "deploy machine learning model to production",
            
            # Conversational data science queries
            "We're contemplating using LDA or NMF for topic modeling. Which is better?",
            "As a data scientist, how do you approach feature selection?",
            "Our team needs to implement A/B testing. What's your methodology?",
            "We have this dataset with missing values. How to handle it?",
            "Being an ML engineer, what's your approach to model deployment?",
            "We're comparing deep learning frameworks. TensorFlow vs PyTorch?",
            "Our startup needs to build a recommendation engine. Where to start?",
            "We have imbalanced data. What techniques do you recommend?",
            "As an AI researcher, how do you evaluate model performance?",
            "We're implementing computer vision. What preprocessing steps?",
            
            # Computer vision and image processing
            "segment cells in microscopy images using deep learning",
            "calculate intersection over union for object detection model",
            "optimize IOU metric for medical image segmentation",
            "detect and count cells in tissue samples",
            "implement semantic segmentation for biomedical images",
            "find distance between objects in image using OpenCV",
            "optimize computer vision model performance",
            "implement image preprocessing pipeline for analysis",
            "create cell tracking algorithm for time-lapse imaging",
            "develop automated tissue analysis system",
            
            # Systems queries
            "configure load balancer for high availability",
            "implement caching strategy using Redis",
            "optimize database performance and indexing",
            "set up monitoring and alerting system",
            "configure CI/CD pipeline using Jenkins",
            "implement distributed system architecture",
            "troubleshoot network connectivity issues",
            "optimize application performance and scalability",
            "implement security best practices for web application",
            "configure cloud infrastructure on AWS"
        ]
    
    def _generate_scientific_examples(self) -> List[str]:
        """Generate comprehensive scientific query examples"""
        return [
            # Mathematical queries - now with conversational framing
            "calculate the derivative of exponential function",
            "solve system of linear equations using matrix methods",
            "find eigenvalues and eigenvectors of matrix",
            "compute integral using numerical integration methods",
            "analyze convergence of infinite series",
            "prove mathematical theorem using induction",
            "optimize function subject to constraints",
            "calculate probability distribution parameters",
            "perform statistical hypothesis testing",
            "analyze correlation between variables",
            
            # Conversational scientific queries
            "We have this article about DDPM. How does it differ from stable diffusion?",
            "As a researcher, what's your take on DDIM vs other diffusion methods?",
            "Our team is studying neural networks. Can you explain backpropagation?",
            "We're analyzing experimental data. Which statistical test to use?",
            "Being a statistician, how do you handle multiple comparisons?",
            "We have this research on climate models. How to validate results?",
            "Our lab needs to design experiments. What's your methodology?",
            "We're comparing different optimization algorithms. Which performs better?",
            "As a physicist, how do you approach uncertainty quantification?",
            "Our research involves Bayesian inference. Best practices?",
            
            # Scientific research queries
            "design controlled experiment to test hypothesis",
            "analyze experimental data using statistical methods",
            "conduct literature review on climate change research",
            "investigate economic benefits of renewable energy",
            "study correlation between education and income",
            "research methodology for social science studies",
            "analyze trends in scientific publication data",
            "evaluate effectiveness of medical treatment",
            "investigate environmental impact of industrial processes",
            "study genetic factors in disease susceptibility",
            
            # Data analysis queries
            "perform exploratory data analysis on customer dataset",
            "conduct A/B testing for website optimization",
            "analyze survey data using regression analysis",
            "investigate patterns in financial market data",
            "study demographic trends in population data",
            "analyze performance metrics and KPIs",
            "investigate causal relationships in observational data",
            "perform time series analysis and forecasting",
            "analyze experimental results for statistical significance",
            "investigate factors affecting business performance"
        ]
    
    def _generate_educational_examples(self) -> List[str]:
        """Generate comprehensive educational query examples"""
        return [
            # Explanation queries
            "explain quantum mechanics concepts in simple terms",
            "teach me about photosynthesis process in plants",
            "what are the causes of World War II",
            "explain how machine learning algorithms work",
            "describe the structure of DNA and its function",
            "explain economic principles of supply and demand",
            "teach me about the water cycle and its importance",
            "what are the benefits of renewable energy sources",
            "explain the concept of natural selection",
            "describe how the human brain processes information",
            
            # Learning support queries
            "help me understand calculus fundamentals",
            "create study guide for biology exam",
            "explain the difference between mitosis and meiosis",
            "teach me about chemical bonding in molecules",
            "help me prepare for physics test on thermodynamics",
            "explain historical significance of Renaissance period",
            "create lesson plan for teaching fractions",
            "help me understand programming concepts",
            "explain the principles of effective communication",
            "teach me about financial literacy and budgeting",
            
            # Academic research queries
            "summarize main points of research paper on climate change",
            "explain methodology used in psychological studies",
            "describe advantages of different research methods",
            "help me understand statistical analysis techniques",
            "explain the peer review process in academic publishing",
            "describe ethical considerations in research",
            "help me write literature review for thesis",
            "explain how to conduct systematic review",
            "describe different types of research designs",
            "help me understand academic writing conventions"
        ]
    
    def _generate_creative_examples(self) -> List[str]:
        """Generate comprehensive creative query examples"""
        return [
            # Creative writing
            "write a short story about time travel",
            "create a poem about autumn leaves",
            "develop character backstory for fantasy novel",
            "write dialogue for dramatic scene",
            "create compelling opening paragraph for mystery story",
            "develop plot outline for science fiction story",
            "write descriptive passage about mountain landscape",
            "create rhyming poem about friendship",
            "develop unique setting for adventure story",
            "write emotional monologue for theater",
            
            # Design and arts
            "design logo for coffee shop brand",
            "create color palette for modern website",
            "develop visual identity for startup company",
            "design poster for music festival",
            "create user interface for mobile app",
            "develop branding strategy for new product",
            "design infographic about environmental issues",
            "create artistic composition using geometric shapes",
            "develop creative concept for advertising campaign",
            "design layout for magazine article",
            
            # Creative brainstorming
            "brainstorm creative solutions for urban transportation",
            "generate innovative ideas for team building activities",
            "develop unique concept for restaurant theme",
            "create original game mechanics for board game",
            "brainstorm creative marketing campaigns",
            "generate ideas for community art project",
            "develop unique approach to employee engagement",
            "create innovative educational activities",
            "brainstorm creative ways to reduce waste",
            "generate ideas for interactive museum exhibit"
        ]
    
    def _generate_business_examples(self) -> List[str]:
        """Generate comprehensive business query examples"""
        return [
            # Business strategy
            "develop business plan for new startup",
            "create marketing strategy for product launch",
            "analyze competitive landscape and market positioning",
            "develop pricing strategy for subscription service",
            "create financial projections for business expansion",
            "design organizational structure for growing company",
            "develop customer acquisition strategy",
            "create employee retention program",
            "analyze market trends and opportunities",
            "develop partnership strategy for business growth",
            
            # Professional communication
            "write professional email to potential client",
            "create presentation for board meeting",
            "draft proposal for new project",
            "write job description for software engineer",
            "create performance review feedback",
            "draft contract terms for service agreement",
            "write professional recommendation letter",
            "create agenda for team meeting",
            "draft press release for product announcement",
            "write executive summary for business report",
            
            # Legal and compliance
            "understand legal requirements for business registration",
            "draft terms of service for website",
            "create privacy policy for mobile app",
            "understand intellectual property protection",
            "draft non-disclosure agreement for partnership",
            "understand employment law compliance",
            "create data protection policy",
            "understand contract law principles",
            "draft licensing agreement for software",
            "understand regulatory compliance requirements"
        ]
    
    def _generate_conversational_examples(self) -> List[str]:
        """Generate comprehensive conversational query examples"""
        return [
            # Personal advice
            "what should I do about relationship problems",
            "how to deal with stress at work",
            "give me advice on career change decisions",
            "help me choose between two job offers",
            "what are good strategies for work-life balance",
            "how to improve communication with colleagues",
            "advice on managing personal finances",
            "help me decide on living situation",
            "what should I consider when buying a car",
            "advice on maintaining healthy relationships",
            
            # Lifestyle and recommendations
            "recommend good books for summer reading",
            "suggest healthy meal ideas for busy schedule",
            "what are good exercises for beginners",
            "recommend travel destinations for families",
            "suggest hobbies for creative expression",
            "what are good ways to learn new skills",
            "recommend strategies for better sleep",
            "suggest ways to stay motivated",
            "what are good practices for mental health",
            "recommend ways to build social connections",
            
            # Opinion and discussion
            "what's your opinion on remote work trends",
            "discuss pros and cons of social media",
            "what do you think about current technology trends",
            "share thoughts on environmental conservation",
            "discuss benefits of lifelong learning",
            "what are your views on work-life integration",
            "discuss importance of cultural diversity",
            "share opinion on digital privacy concerns",
            "discuss impact of artificial intelligence",
            "what are your thoughts on sustainable living"
        ]
    
    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query using enhanced embedding approach"""
        logger.info(f"Starting enhanced embedding classification for: '{query[:100]}...'")
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities with all categories
        similarities = {}
        for category, cat_embedding in self.category_embeddings.items():
            similarity = cosine_similarity([query_embedding], [cat_embedding])[0][0]
            similarities[category] = similarity
            logger.info(f"Similarity with {category}: {similarity:.4f}")
        
        # Find best match
        best_category = max(similarities, key=similarities.get)
        confidence = similarities[best_category]
        
        # Apply confidence adjustment
        if confidence > 0.85:
            confidence = min(confidence + 0.1, 1.0)
        
        logger.info(f"Enhanced embedding classification result: '{best_category}' with confidence {confidence:.4f}")
        
        return best_category, confidence

class SyntheticQueryGenerator:
    """Generates synthetic training queries for better category representation"""
    
    def __init__(self):
        self.domain_templates = {
            'CODE_TECHNICAL': [
                "how to {action} {technology} {object}",
                "debug {error} in {technology}",
                "implement {algorithm} using {language}",
                "optimize {component} for {goal}",
                "create {artifact} with {framework}"
            ],
            'MATHEMATICAL_SCIENTIFIC': [
                "calculate {mathematical_concept} of {object}",
                "analyze {data_type} using {method}",
                "investigate {phenomenon} in {domain}",
                "study {relationship} between {variables}",
                "research {topic} methodology"
            ]
            # Add more templates as needed
        }
    
    def generate_synthetic_queries(self, category: str, num_queries: int = 50) -> List[str]:
        """Generate synthetic queries for a specific category"""
        # Implementation would use templates and domain-specific vocabularies
        # to generate diverse synthetic queries
        pass

# Usage example and integration point
class EnhancedQueryClassificationSystem:
    """Main system combining enhanced rule-based and embedding approaches"""
    
    def __init__(self):
        self.rule_classifier = EnhancedRuleBasedClassifier()
        self.embedding_classifier = EnhancedEmbeddingClassifier()
        
    def classify(self, query: str) -> Tuple[str, float, Dict]:
        """Classify query using ensemble of enhanced methods"""
        
        # Get classifications from both methods
        rule_category, rule_confidence = self.rule_classifier.classify(query)
        embedding_category, embedding_confidence = self.embedding_classifier.classify(query)
        
        # Ensemble decision making
        if rule_confidence > 0.8 and embedding_confidence > 0.8:
            if rule_category == embedding_category:
                # High agreement - boost confidence
                final_category = rule_category
                final_confidence = min((rule_confidence + embedding_confidence) / 2 + 0.1, 1.0)
            else:
                # High confidence but disagreement - use higher confidence
                if rule_confidence > embedding_confidence:
                    final_category, final_confidence = rule_category, rule_confidence
                else:
                    final_category, final_confidence = embedding_category, embedding_confidence
        elif rule_confidence > embedding_confidence:
            final_category, final_confidence = rule_category, rule_confidence
        else:
            final_category, final_confidence = embedding_category, embedding_confidence
        
        # Return detailed results
        results = {
            'final_category': final_category,
            'final_confidence': final_confidence,
            'rule_based': {'category': rule_category, 'confidence': rule_confidence},
            'embedding_based': {'category': embedding_category, 'confidence': embedding_confidence}
        }
        
        return final_category, final_confidence, results

if __name__ == "__main__":
    # Example usage
    system = EnhancedQueryClassificationSystem()
    
    test_queries = [
        # "write a python function to sort a list of numbers",
        # "what are the benefits of renewable energy in economics",
        # "explain quantum mechanics in simple terms",
        # "create a short story about time travel",
        # "develop business plan for new startup",
        # "give me advice on career change",
        # "How to reduce energy consumption in my home?",
        # "What is the best way to learn programming?",
        # "How to analyze time series data using LSTM networks?",
        # "What are the key principles of effective communication?",
        # "What are eigenvalues and eigenvectors of a matrix?",
        # "What's TLS with respect to biology?",
        # "Write the formula for KL divergence",
        # "how to find distance of cells in a tissue and minimize IOU how to do it?",
        # "What's the formula for IOU and how to calculate it from data?"
        "You're a senior software engineer. How do I debug this React component in the following code?",
        "We have this article about DDPM. We want to differentiate this method fro mstable diffusion and DDIM, be concise",
        "We're contemplating using LDA or NMF for topic modeling. Which one is better for our dataset?",
    ]
    
    for query in test_queries:
        category, confidence, details = system.classify(query)
        print(f"Query: {query}")
        print(f"Category: {category} (confidence: {confidence:.3f})")
        print(f"Details: {details}")
        print("-" * 50)
