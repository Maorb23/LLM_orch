"""
Integration Example: Enhancing Your Existing QueryPreprocessor and QueryClassifier
This shows how to practically integrate the enhanced NLP features into your current system
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

# Safe NLTK import with installation guidance
try:
    import nltk
    print("NLTK found, proceeding with setup...")
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not found. Please install it with: pip install nltk")
    print("Falling back to basic text processing...")
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    # Download punkt_tab first for tokenization
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

if NLTK_AVAILABLE:
    # Download required NLTK data (run once)
    def ensure_nltk_data():
        """Download required NLTK data if not present"""
        # Define required NLTK data with their categories
        required_data = {
            # Tokenizers - Essential for text processing
            'punkt': 'tokenizers',
            'punkt_tab': 'tokenizers',  # For newer NLTK versions
            
            # Corpora - Language data
            'stopwords': 'corpora',
            'wordnet': 'corpora', 
            'omw-1.4': 'corpora',  # Open Multilingual Wordnet
            'words': 'corpora',
            'brown': 'corpora',  # Additional language modeling
            
            # Taggers - Part-of-speech tagging
            'averaged_perceptron_tagger': 'taggers',
            'averaged_perceptron_tagger_eng': 'taggers',  # For newer NLTK versions
            
            # Chunkers - Named entity recognition
            'maxent_ne_chunker': 'chunkers',
            
            # Additional useful data
            'vader_lexicon': 'corpora',  # For sentiment analysis (optional)
        }
        
        print("Checking and downloading required NLTK data...")
        downloaded_packages = []
        failed_packages = []
        
        for data_name, category in required_data.items():
            try:
                # Try to find the data in the expected category
                nltk.data.find(f'{category}/{data_name}')
                print(f"✓ {data_name} already available")
            except LookupError:
                try:
                    print(f"Downloading {data_name}...", end=" ")
                    nltk.download(data_name, quiet=True)
                    downloaded_packages.append(data_name)
                    print("✓")
                except Exception as e:
                    print(f"⚠ Warning: Could not download {data_name}: {e}")
                    failed_packages.append(data_name)
                    continue
        
        # Summary
        if downloaded_packages:
            print(f"Downloaded {len(downloaded_packages)} new packages: {', '.join(downloaded_packages)}")
        
        if failed_packages:
            print(f"Failed to download {len(failed_packages)} packages: {', '.join(failed_packages)}")
            print("Note: Some advanced features may not work, but basic functionality should be available.")
        
        print("NLTK data setup completed!")
        return len(failed_packages) < len(required_data) // 2  # Success if less than half failed
    
    # Test basic NLTK functionality
    def test_nltk_functionality():
        """Test that essential NLTK components are working"""
        try:
            # Test tokenization
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize("Test sentence.")
            
            # Test stopwords
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
            
            # Test stemming
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            stemmed = stemmer.stem("running")
            
            print("✓ NLTK core functionality verified")
            return True
            
        except Exception as e:
            print(f"⚠ NLTK functionality test failed: {e}")
            return False
else:
    def ensure_nltk_data():
        """Fallback when NLTK is not available"""
        print("NLTK not available - install with: pip install nltk")
        return False
    
    def test_nltk_functionality():
        """Fallback test when NLTK is not available"""
        print("NLTK not available for testing")
        return False

# Initialize NLTK setup
if NLTK_AVAILABLE:
    print("Setting up NLTK data...")
    nltk_success = ensure_nltk_data()
    if nltk_success:
        test_nltk_functionality()
    else:
        print("⚠ NLTK setup had issues - some features may not work")
        NLTK_AVAILABLE = False
else:
    ensure_nltk_data()  # Show installation message

# Conditional NLTK imports
if NLTK_AVAILABLE:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
else:
    # Fallback functions when NLTK is not available
    def word_tokenize(text):
        """Basic tokenization fallback"""
        return text.lower().split()
    
    def pos_tag(tokens):
        """Basic POS tagging fallback"""
        return [(token, 'NN') for token in tokens]
    
    def ne_chunk(pos_tags):
        """Basic NER fallback"""
        return []
    
    class PorterStemmer:
        def stem(self, word):
            return word
    
    class WordNetLemmatizer:
        def lemmatize(self, word, pos='n'):
            return word
    
    stopwords = type('stopwords', (), {'words': lambda lang: set()})()
    wordnet = type('wordnet', (), {'synsets': lambda word: []})()

logger = logging.getLogger(__name__)

class EnhancedDomainLexicons:
    """Enhanced domain-specific lexicons using NLTK WordNet expansion"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Your existing technical terms + enhanced versions
        self.base_lexicons = {
            'technical': [
                # Programming core
                'algorithm', 'array', 'binary', 'class', 'compile', 'debug', 'function',
                'inheritance', 'loop', 'method', 'object', 'parameter', 'recursion',
                'syntax', 'variable', 'framework', 'library', 'module', 'package',
                'repository', 'version', 'branch', 'commit', 'merge', 'pull', 'push',
                'python', 'javascript', 'java', 'cpp', 'programming', 'code', 'coding',
                
                # Data Science
                'tensorflow', 'pytorch', 'numpy', 'pandas', 'sklearn', 'matplotlib',
                'neural', 'network', 'machine', 'learning', 'deep', 'model', 'training',
                'dataset', 'feature', 'prediction', 'classification', 'regression',
                
                # Web/Systems
                'api', 'rest', 'graphql', 'database', 'sql', 'mysql', 'postgresql',
                'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 'cloud',
                'microservice', 'deployment', 'scaling', 'performance'
            ],
            
            'business': [
                # Finance
                'revenue', 'profit', 'budget', 'investment', 'capital', 'expense',
                'income', 'cash', 'flow', 'balance', 'financial', 'accounting',
                'roi', 'cost', 'pricing', 'valuation', 'funding', 'venture',
                
                # Marketing/Strategy  
                'marketing', 'strategy', 'business', 'customer', 'client', 'market',
                'campaign', 'brand', 'segment', 'target', 'audience', 'conversion',
                'analytics', 'metrics', 'kpi', 'growth', 'acquisition', 'retention',
                
                # Management/Legal
                'management', 'leadership', 'team', 'project', 'planning', 'execution',
                'stakeholder', 'communication', 'legal', 'contract', 'compliance',
                'regulation', 'policy', 'governance', 'risk', 'audit'
            ],
            
            'academic': [
                # Research/Education
                'research', 'study', 'analysis', 'hypothesis', 'methodology',
                'experiment', 'observation', 'data', 'statistics', 'correlation',
                'education', 'learning', 'teaching', 'academic', 'university',
                'student', 'professor', 'course', 'curriculum', 'assessment',
                
                # Scientific
                'scientific', 'science', 'theory', 'principle', 'law', 'equation',
                'formula', 'calculation', 'measurement', 'laboratory', 'evidence',
                'peer', 'review', 'publication', 'journal', 'conference'
            ],
            
            'creative': [
                # Writing/Content
                'creative', 'writing', 'story', 'narrative', 'character', 'plot',
                'theme', 'dialogue', 'poetry', 'poem', 'prose', 'fiction',
                'content', 'blog', 'article', 'copywriting',
                
                # Design/Arts
                'design', 'graphic', 'visual', 'art', 'artistic', 'aesthetic',
                'color', 'typography', 'layout', 'composition', 'illustration',
                'branding', 'logo', 'identity', 'creative', 'concept',
                'music', 'song', 'melody', 'rhythm', 'performance'
            ],
            
            'scientific': [
                # Mathematics
                'mathematics', 'mathematical', 'equation', 'formula', 'calculation',
                'algebra', 'calculus', 'geometry', 'statistics', 'probability',
                'derivative', 'integral', 'matrix', 'vector', 'optimization',
                
                # Research Methods
                'methodology', 'experiment', 'hypothesis', 'analysis', 'correlation',
                'regression', 'significance', 'p-value', 'confidence', 'interval',
                'sample', 'population', 'distribution', 'variance', 'deviation'
            ]
        }
        
        # Expand lexicons using WordNet
        self.expanded_lexicons = self._expand_with_wordnet()
        
    def _expand_with_wordnet(self) -> Dict[str, Set[str]]:
        """Expand each domain lexicon using WordNet synsets"""
        logger.info("Expanding domain lexicons with WordNet synsets...")
        
        expanded = {}
        
        for domain, base_terms in self.base_lexicons.items():
            expanded_terms = set()
            
            for term in base_terms:
                # Add original term
                expanded_terms.add(term)
                
                # Add morphological variations
                expanded_terms.add(term + 's')      # plural
                expanded_terms.add(term + 'ing')    # gerund
                if not term.endswith('e'):
                    expanded_terms.add(term + 'ed') # past tense
                else:
                    expanded_terms.add(term + 'd')   # past tense for -e ending
                
                # Add stem and lemma
                expanded_terms.add(self.stemmer.stem(term))
                expanded_terms.add(self.lemmatizer.lemmatize(term))
                expanded_terms.add(self.lemmatizer.lemmatize(term, pos='v'))  # verb form
                
                # Add WordNet synsets (synonyms)
                synsets = wordnet.synsets(term)
                for synset in synsets[:2]:  # Limit to top 2 synsets to avoid noise
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        if len(synonym) > 2 and synonym.isalpha():
                            expanded_terms.add(synonym)
            
            # Filter out stop words and very short terms
            expanded_terms = {
                term for term in expanded_terms 
                if len(term) > 2 and term not in self.stop_words
            }
            
            expanded[domain] = expanded_terms
            logger.info(f"Domain '{domain}': {len(base_terms)} → {len(expanded_terms)} terms")
        
        return expanded
    
    def get_domain_terms(self, domain: str) -> Set[str]:
        """Get expanded terms for a domain"""
        return self.expanded_lexicons.get(domain, set())
    
    def calculate_domain_density(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate domain-specific term density"""
        if not tokens:
            return {domain: 0.0 for domain in self.expanded_lexicons.keys()}
        
        token_set = set(token.lower() for token in tokens)
        densities = {}
        
        for domain, domain_terms in self.expanded_lexicons.items():
            matches = len(token_set.intersection(domain_terms))
            densities[domain] = matches / len(tokens)
        
        return densities

class EnhancedQueryPreprocessor:
    """Enhanced version of your QueryPreprocessor with advanced NLP"""
    
    def __init__(self):
        logger.info("Initializing EnhancedQueryPreprocessor with NLTK")
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced domain lexicons
        self.domain_lexicons = EnhancedDomainLexicons()
        
        # Your existing tech terms (keep for compatibility)
        self.shortened_tech_terms = {
            'js': 'javascript', 'py': 'python', 'ts': 'typescript',
            'cpp': 'c++', 'cs': 'c#', 'rb': 'ruby', 'php': 'php',
            'ai': 'artificial intelligence', 'ml': 'machine learning',
            'dl': 'deep learning', 'nlp': 'natural language processing',
            'db': 'database', 'sql': 'structured query language',
            'api': 'application programming interface',
            'tf': 'tensorflow', 'np': 'numpy', 'pd': 'pandas'
        }
        
        # Enhanced intent patterns
        self.intent_patterns = {
            'how_to': r'\b(how\s+to|how\s+do\s+i|how\s+can\s+i|how\s+should\s+i)\b',
            'what_is': r'\b(what\s+is|what\s+are|what\s+does|what\s+means?)\b',
            'why': r'\b(why\s+is|why\s+do|why\s+does|why\s+are|why\s+would)\b',
            'explain': r'\b(explain|describe|tell\s+me\s+about|clarify)\b',
            'create': r'\b(create|build|make|generate|write|develop|design)\b',
            'fix': r'\b(fix|debug|solve|resolve|troubleshoot|repair)\b',
            'compare': r'\b(compare|difference|versus|vs|contrast|better|worse)\b',
            'analyze': r'\b(analyze|examine|study|investigate|evaluate|assess)\b',
            'benefits': r'\b(benefits|advantages|pros|positive|good|help)\b',
            'implement': r'\b(implement|code|program|build|construct)\b'
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            'technical_jargon': r'\b(implementation|architecture|optimization|scalability|algorithm|framework)\b',
            'academic_language': r'\b(hypothesis|methodology|empirical|theoretical|systematic|research)\b',
            'formal_language': r'\b(furthermore|consequently|nevertheless|accordingly|therefore|thus)\b',
            'code_elements': r'[{}()\[\];]|def\s+|class\s+|import\s+|function\s*\(|\w+\.\w+\(',
            'mathematical': r'[∑∏∫∂∇=<>≤≥±]|\\sum|\\int|\\frac|equation|formula|theorem|calculate',
            'business_formal': r'\b(proposal|strategy|revenue|roi|stakeholder|compliance|governance)\b'
        }
        
        logger.info("EnhancedQueryPreprocessor initialized successfully")
    
    def extract_enhanced_features(self, query: str) -> Dict:
        """Extract comprehensive features using NLTK and domain analysis"""
        logger.info(f"Extracting enhanced features for: '{query[:50]}...'")
        
        # Basic preprocessing
        query_lower = query.lower().strip()
        
        # Tokenization using NLTK
        tokens = word_tokenize(query_lower)
        tokens = [token for token in tokens if token.isalnum()]  # Keep only alphanumeric
        
        # Content tokens (without stop words)
        content_tokens = [token for token in tokens if token not in self.stop_words]
        
        # Linguistic analysis
        pos_tags = pos_tag(tokens)
        
        # Named Entity Recognition
        try:
            chunks = ne_chunk(pos_tags)
            named_entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    named_entities.append(entity)
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            named_entities = []
        
        # Stemming and lemmatization
        stems = [self.stemmer.stem(token) for token in content_tokens]
        lemmas = [self.lemmatizer.lemmatize(token) for token in content_tokens]
        
        # All word forms for matching
        all_word_forms = set(tokens + content_tokens + stems + lemmas)
        
        # Domain density analysis using enhanced lexicons
        domain_densities = self.domain_lexicons.calculate_domain_density(content_tokens)
        
        # Extract domain-specific terms
        domain_terms = {}
        for domain in ['technical', 'business', 'academic', 'creative', 'scientific']:
            domain_lexicon = self.domain_lexicons.get_domain_terms(domain)
            found_terms = list(all_word_forms.intersection(domain_lexicon))
            
            # Also check for multi-word terms in original query
            for term in domain_lexicon:
                if ' ' in term and term in query_lower:
                    found_terms.append(term)
            
            domain_terms[f'{domain}_terms'] = list(set(found_terms))
        
        # Intent pattern matching
        detected_intents = []
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                detected_intents.append(intent)
        
        # Complexity analysis
        complexity_scores = {}
        for complexity_type, pattern in self.complexity_patterns.items():
            matches = len(re.findall(pattern, query_lower))
            complexity_scores[complexity_type] = matches / len(tokens) if tokens else 0
        
        # Question type detection
        question_type = self._detect_question_type(query_lower)
        
        # Statistical features
        lexical_diversity = len(set(content_tokens)) / len(content_tokens) if content_tokens else 0
        avg_word_length = sum(len(token) for token in content_tokens) / len(content_tokens) if content_tokens else 0
        
        # Part-of-speech distribution
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_pos = sum(pos_counts.values())
        pos_distribution = {pos: count/total_pos for pos, count in pos_counts.items()} if total_pos > 0 else {}
        
        # Compile enhanced features
        enhanced_features = {
            # Basic features (maintain compatibility)
            'tokens': tokens,
            'word_count': len(tokens),
            'char_count': len(query),
            
            # Linguistic features
            'content_tokens': content_tokens,
            'stems': stems,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'pos_distribution': pos_distribution,
            'named_entities': named_entities,
            
            # Domain analysis
            'domain_densities': domain_densities,
            **domain_terms,  # technical_terms, business_terms, etc.
            
            # Intent and complexity
            'detected_intents': detected_intents,
            'question_type': question_type,
            'complexity_scores': complexity_scores,
            
            # Statistical measures
            'lexical_diversity': lexical_diversity,
            'avg_word_length': avg_word_length,
            'has_numbers': bool(re.search(r'\d', query)),
            'has_special_chars': bool(re.search(r'[<>@#$%^&*]', query)),
            'has_code_markers': bool(re.search(r'[{}()\[\];]', query)),
            
            # Legacy compatibility
            'tech_keywords': domain_terms.get('technical_terms', []),
            'has_tech_terms': len(domain_terms.get('technical_terms', [])) > 0,
            'tech_term_count': len(domain_terms.get('technical_terms', [])),
        }
        
        logger.info(f"Enhanced features extracted:")
        logger.info(f"  Domain densities: {domain_densities}")
        logger.info(f"  Detected intents: {detected_intents}")
        logger.info(f"  Question type: {question_type}")
        logger.info(f"  Complexity scores: {complexity_scores}")
        
        return enhanced_features
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked"""
        question_patterns = {
            'factual': r'^(what|when|where|who|which)\b',
            'procedural': r'^(how|how\s+to|how\s+do|how\s+can)\b',
            'causal': r'^(why|what\s+causes|what\s+makes)\b',
            'comparative': r'\b(better|worse|difference|compare|versus|vs|contrast)\b',
            'evaluative': r'\b(should|would|could|recommend|suggest|opinion|best|worst)\b',
            'creative': r'\b(create|write|design|compose|generate|make)\b',
            'analytical': r'\b(analyze|examine|study|investigate|research|evaluate)\b',
            'benefits': r'\b(benefits|advantages|pros|positive|good|help|useful)\b'
        }
        
        for q_type, pattern in question_patterns.items():
            if re.search(pattern, query):
                return q_type
        
        return 'general'
    
    def process(self, query: str):
        """Enhanced version of your process method"""
        logger.info(f"Processing query with enhanced NLP: '{query[:50]}...'")
        
        try:
            # Extract enhanced features
            enhanced_features = self.extract_enhanced_features(query)
            
            # Maintain compatibility with existing ProcessedQuery structure
            from your_existing_module import ProcessedQuery  # Import your existing class
            
            processed_query = ProcessedQuery(
                original_query=query,
                normalized_query=self.normalize_text(query),  # Use your existing method
                key_terms=enhanced_features.get('tech_keywords', []),
                action_verbs=enhanced_features.get('detected_intents', []),
                features=enhanced_features
            )
            
            return processed_query
            
        except Exception as e:
            logger.warning(f"Enhanced processing failed: {e}, falling back to basic processing")
            # Fallback to your existing process method
            return super().process(query)

class EnhancedRuleClassifier:
    """Enhanced rule-based classifier using comprehensive NLP features"""
    
    def __init__(self):
        self.preprocessor = EnhancedQueryPreprocessor()
        
        # Enhanced classification rules with multiple criteria
        self.classification_rules = {
            'CODE_TECHNICAL': {
                'domain_density_threshold': {'technical': 0.25},
                'required_intents': ['create', 'fix', 'how_to', 'implement'],
                'required_question_types': ['procedural', 'creative', 'analytical'],
                'complexity_indicators': ['technical_jargon', 'code_elements'],
                'pos_preferences': {'VB': 0.15, 'NN': 0.30},  # Verbs and nouns
                'negative_domains': {'creative': 0.20, 'business': 0.30}
            },
            'MATHEMATICAL_SCIENTIFIC': {
                'domain_density_threshold': {'scientific': 0.20, 'academic': 0.15},
                'required_intents': ['analyze', 'explain', 'what_is', 'benefits'],
                'required_question_types': ['factual', 'analytical', 'causal', 'benefits'],
                'complexity_indicators': ['mathematical', 'academic_language'],
                'pos_preferences': {'NN': 0.35, 'JJ': 0.20},
                'negative_domains': {'creative': 0.25, 'business': 0.15}
            },
            'EDUCATIONAL_ACADEMIC': {
                'domain_density_threshold': {'academic': 0.20},
                'required_intents': ['explain', 'what_is', 'why', 'how_to', 'benefits'],
                'required_question_types': ['factual', 'procedural', 'causal', 'benefits'],
                'complexity_indicators': ['academic_language', 'formal_language'],
                'pos_preferences': {'NN': 0.30, 'VB': 0.20},
                'negative_domains': {'business': 0.20}
            },
            'CREATIVE_ARTISTIC': {
                'domain_density_threshold': {'creative': 0.20},
                'required_intents': ['create', 'how_to'],
                'required_question_types': ['creative', 'procedural'],
                'complexity_indicators': [],
                'pos_preferences': {'JJ': 0.25, 'RB': 0.15},  # Adjectives and adverbs
                'negative_domains': {'technical': 0.30, 'business': 0.30, 'scientific': 0.30}
            },
            'BUSINESS_PROFESSIONAL': {
                'domain_density_threshold': {'business': 0.20},
                'required_intents': ['create', 'analyze', 'compare', 'explain'],
                'required_question_types': ['procedural', 'analytical', 'evaluative'],
                'complexity_indicators': ['business_formal', 'formal_language'],
                'pos_preferences': {'NN': 0.35, 'VB': 0.20},
                'negative_domains': {'creative': 0.30}
            },
            'CONVERSATIONAL_ADVICE': {
                'domain_density_threshold': {},  # No specific domain required
                'required_intents': ['how_to', 'what_is', 'why'],
                'required_question_types': ['evaluative', 'procedural', 'general'],
                'complexity_indicators': [],
                'pos_preferences': {},
                'negative_domains': {'technical': 0.20, 'business': 0.20, 'academic': 0.20, 'scientific': 0.20}
            }
        }
    
    def classify(self, query: str) -> Tuple[str, float]:
        """Enhanced rule-based classification"""
        logger.info(f"Enhanced rule-based classification for: '{query[:50]}...'")
        
        # Extract enhanced features
        features = self.preprocessor.extract_enhanced_features(query)
        
        # Calculate scores for each category
        category_scores = {}
        
        for category, rules in self.classification_rules.items():
            score = self._calculate_enhanced_score(features, rules)
            category_scores[category] = score
            logger.info(f"Category '{category}' enhanced score: {score:.4f}")
        
        # Find best category
        if not category_scores or max(category_scores.values()) < 0.1:
            return 'CONVERSATIONAL_ADVICE', 0.15
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category], 1.0)
        
        logger.info(f"Enhanced rule-based result: '{best_category}' with confidence {confidence:.4f}")
        
        return best_category, confidence
    
    def _calculate_enhanced_score(self, features: Dict, rules: Dict) -> float:
        """Calculate category score using enhanced features"""
        score = 0.0
        
        # Domain density scoring
        domain_thresholds = rules.get('domain_density_threshold', {})
        domain_densities = features.get('domain_densities', {})
        
        for domain, threshold in domain_thresholds.items():
            actual_density = domain_densities.get(domain, 0)
            if actual_density >= threshold:
                score += 0.3  # Base score for meeting threshold
                score += (actual_density - threshold) * 0.5  # Bonus for exceeding
        
        # Intent matching
        required_intents = set(rules.get('required_intents', []))
        detected_intents = set(features.get('detected_intents', []))
        intent_matches = len(required_intents.intersection(detected_intents))
        if intent_matches > 0:
            score += 0.2 + (intent_matches - 1) * 0.1  # Bonus for multiple matches
        
        # Question type matching
        required_types = set(rules.get('required_question_types', []))
        question_type = features.get('question_type', '')
        if question_type in required_types:
            score += 0.2
        
        # Complexity indicators
        complexity_indicators = rules.get('complexity_indicators', [])
        complexity_scores = features.get('complexity_scores', {})
        for indicator in complexity_indicators:
            if indicator in complexity_scores and complexity_scores[indicator] > 0:
                score += complexity_scores[indicator] * 0.15
        
        # POS preferences
        pos_preferences = rules.get('pos_preferences', {})
        pos_distribution = features.get('pos_distribution', {})
        for pos, preferred_ratio in pos_preferences.items():
            actual_ratio = pos_distribution.get(pos, 0)
            if actual_ratio >= preferred_ratio:
                score += 0.1
        
        # Negative domain penalties
        negative_domains = rules.get('negative_domains', {})
        for domain, penalty_threshold in negative_domains.items():
            actual_density = domain_densities.get(domain, 0)
            if actual_density > penalty_threshold:
                penalty = (actual_density - penalty_threshold) * 0.4
                score -= penalty
        
        return max(0.0, score)

# Example integration with your existing system
def integrate_enhanced_classification():
    """Example of how to integrate the enhanced system"""
    
    # Create enhanced preprocessor
    enhanced_preprocessor = EnhancedQueryPreprocessor()
    enhanced_classifier = EnhancedRuleClassifier()
    
    # Test queries
    test_queries = [
        "write a python function to sort a list using quicksort algorithm",
        "what are the economic benefits of renewable energy investments",
        "explain quantum mechanics principles in simple terms for students",
        "create a short story about artificial intelligence becoming sentient",
        "develop a comprehensive marketing strategy for startup growth",
        "give me advice on choosing between two job offers"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Enhanced processing
        enhanced_features = enhanced_preprocessor.extract_enhanced_features(query)
        category, confidence = enhanced_classifier.classify(query)
        
        print(f"Classification: {category} (confidence: {confidence:.3f})")
        print(f"Domain densities: {enhanced_features['domain_densities']}")
        print(f"Detected intents: {enhanced_features['detected_intents']}")
        print(f"Question type: {enhanced_features['question_type']}")
        print("-" * 70)

if __name__ == "__main__":
    integrate_enhanced_classification()
