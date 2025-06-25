from classify_query import LLMQueryRouter
import argparse

def test_absorbed_categories():
    router = LLMQueryRouter("Nebius_api_key.txt")
    
    test_queries = [
        "What are Eigenvalues and Eigenvectors?",
        "Using NN what are the biases and weights?",
        "Who wrote the book 'The Great Gatsby'? and what is its main theme?",
        "Explain the concept of Quantum Entanglement in simple terms.",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = router.process_query(query)
        
        # DEBUG: Print the entire result to see what's returned
        print(f"DEBUG - Full result: {result}")
        print(f"DEBUG - Result keys: {list(result.keys())}")
        
        # Check if processing was successful
        if not result.get('success', False):
            print(f"❌ ERROR: {result.get('error', 'Unknown error')}")
            print(f"💬 Error Response: {result.get('response', 'No response')}")
            continue
        
        # Safe extraction with error handling
        classification = result.get('classification', {})
        if not classification:
            print("❌ ERROR: No classification data returned")
            continue
            
        print(f"✅ Primary Category: {classification.get('category', 'UNKNOWN')}")
        print(f"✅ Confidence: {classification.get('confidence', 0):.3f}")
        print(f"✅ Response: {result.get('response', 'No response')[:200]}...")

from archive.classify_query_old import QueryPreprocessor

def test_enhanced_tech_terms():
    preprocessor = QueryPreprocessor()
    
    test_queries = [
        # Test normalization
        "Write a JS function using ML algorithms",
        "How to use tf for deep learning with py?",
        "Build an API with node.js and nosql db",
        
        # Test preservation
        "Explain machine learning vs deep learning",
        "React vs Vue.js for frontend development",
        "Python pandas dataframe operations",
        
        # Test multi-word terms
        "What is natural language processing in AI?",
        "How does convolutional neural network work?",
        "Implement continuous integration with Jenkins",
        
        # Test mixed cases
        "Use TensorFlow and PyTorch for ML model training",
        "REST API vs GraphQL for web development",
        "Docker containers in Kubernetes cluster"
    ]
    
    print("🔧 Enhanced Technical Terms Test")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: '{query}'")
        
        # Process the query
        processed = preprocessor.process(query)
        
        print(f"   Normalized: '{processed.normalized_query}'")
        print(f"   Tech Keywords Found: {processed.key_terms}")
        print(f"   Action Verbs: {processed.action_verbs}")
        print(f"   Tech Categories: {processed.features.get('tech_categories_found', [])}")
        print(f"   Primary Category: {processed.features.get('primary_tech_category', 'none')}")
        print(f"   Tech Density: {processed.features['complexity']['tech_density']:.2f}")
    
    # Test comprehensive tech terms list
    print(f"\n📊 Technical Terms Statistics:")
    print(f"   Total tech terms: {len(preprocessor.all_tech_terms)}")
    print(f"   Shortened terms: {len(preprocessor.shortened_tech_terms)}")
    print(f"   Preserved terms: {len(preprocessor.preserve_tech_terms)}")
    
    # Show some examples
    print(f"\n🔤 Sample Technical Terms:")
    sample_terms = list(preprocessor.all_tech_terms)[:20]
    for term in sorted(sample_terms):
        print(f"   • {term}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_absorbed_categories", "--ac",
                         action="store_true", help="Test absorbed categories")
    parser.add_argument("--test_enhanced_tech_terms", "--ett",
                         action="store_true", help="Test enhanced technical terms processing")
    args = parser.parse_args()
    if args.test_absorbed_categories:
        test_absorbed_categories()
    if args.test_enhanced_tech_terms:
        test_enhanced_tech_terms()