import json
import re
import random
from typing import Dict, Any, Tuple

# This is a mock client to simulate API calls to different LLMs.
# In a real system, this would interact with actual model endpoints.
class MockLLMClient:
    """
    A mock client to simulate responses from specialized and general LLMs.
    This helps in demonstrating the routing logic without real API calls.
    """
    def generate_response(self, query: str, model_type: str) -> str:
        """Simulates generating a response from a specific model type."""
        return f"Response from '{model_type}' model for query: '{query}'"

    def classify_query(self, features: Dict[str, Any]) -> str:
        """
        Simulates the classification LLM's response.
        In a real scenario, this would be a network request.
        """
        query = features.get("original_query", "")
        if "python" in query or "code" in query:
            category = "Code & Development"
            confidence = 0.95
        elif "story" in query or "poem" in query:
            category = "Creative Writing"
            confidence = 0.92
        elif "help" in query or len(query.split()) < 3:
            category = "Conversational & Chit-Chat"
            confidence = 0.45 # Lower confidence to demonstrate fallback
        else:
            category = "Factual Q&A"
            confidence = 0.88

        # The response is a JSON string, similar to a real LLM API call.
        return json.dumps({
            "category": category,
            "confidence": confidence,
            "justification": "Based on keywords and query structure."
        })


class QueryRouter:
    """
    Encapsulates the logic for classifying a user query and routing it
    to the appropriate Large Language Model (LLM) based on the classification result.
    """

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initializes the QueryRouter.

        Args:
            confidence_threshold (float): The minimum confidence score required
                                          to route to a specialized LLM.
        """
        self.confidence_threshold = confidence_threshold
        # In a real implementation, you would initialize your actual API client here.
        # self.classifier_client = OpenAI(base_url="...", api_key="...")
        self.classifier_client = MockLLMClient()
        self.llm_client = MockLLMClient()


    def _preprocess_query(self, query: str) -> str:
        """
        Performs basic text normalization on the user query.

        Args:
            query (str): The raw user query.

        Returns:
            str: The processed query (e.g., lowercased, extra whitespace removed).
        """
        # A[User Query] --> B[Preprocessing Pipeline]
        print(f"1. Preprocessing query: '{query}'")
        processed_query = query.lower().strip()
        # Remove extra whitespace
        processed_query = re.sub(r'\s+', ' ', processed_query)
        return processed_query

    def _extract_features(self, query: str, processed_query: str) -> Dict[str, Any]:
        """
        Extracts features from the query to feed into the classifier.

        Args:
            query (str): The original user query.
            processed_query (str): The normalized query.

        Returns:
            Dict[str, Any]: A dictionary of features.
        """
        # B[Preprocessing Pipeline] --> C[Feature Extraction]
        print("2. Extracting features...")
        return {
            "original_query": query,
            "processed_query": processed_query,
            "query_length": len(processed_query.split()),
            "has_question_mark": "?" in processed_query
        }

    def _classify_query(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the classification engine to get the query's category and confidence.

        Args:
            features (Dict[str, Any]): The features extracted from the query.

        Returns:
            Dict[str, Any]: A dictionary containing the category and confidence score.
                            Returns a default error response on failure.
        """
        # C[Feature Extraction] --> D[Classification Engine]
        print("3. Classifying features...")
        try:
            # This simulates calling an external LLM for classification
            response_str = self.classifier_client.classify_query(features)
            classification_result = json.loads(response_str)
            return classification_result
        except json.JSONDecodeError:
            print("Error: Failed to decode classification response.")
            return {"category": "unknown", "confidence": 0.0}
        except Exception as e:
            print(f"An unexpected error occurred during classification: {e}")
            return {"category": "unknown", "confidence": 0.0}

    def _postprocess_response(self, response: str) -> str:
        """
        Applies final formatting to the LLM's response.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            str: The formatted response ready for the user.
        """
        # I[Response Generation] --> J[Response Post-processing] --> K[User Response]
        print("6. Post-processing final response.")
        return response.strip() + "\n--- End of Response ---"

    def handle_query(self, query: str) -> str:
        """
        Main method to orchestrate the entire query handling flow.

        Args:
            query (str): The user's input query.

        Returns:
            str: The final, processed response from the appropriate LLM.
        """
        processed_query = self._preprocess_query(query)
        features = self._extract_features(query, processed_query)
        classification = self._classify_query(features)

        category = classification.get("category", "unknown")
        confidence = classification.get("confidence", 0.0)

        print(f"4. Evaluating confidence: {confidence:.2f} (Threshold: {self.confidence_threshold})")
        # E[Confidence Evaluation] --> F{Confidence > Threshold?}
        if confidence >= self.confidence_threshold:
            # F -->|Yes| G[Route to Specialized LLM]
            print(f"--> Routing to SPECIALIZED LLM: '{category}'")
            model_to_use = category # e.g., "Code & Development"
        else:
            # F -->|No| H[Route to General LLM]
            print(f"--> Confidence below threshold. Routing to GENERAL LLM.")
            model_to_use = "General Purpose"

        # G/H --> I[Response Generation]
        print("5. Generating response...")
        final_response = self.llm_client.generate_response(query, model_to_use)

        return self._postprocess_response(final_response)


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the router with a confidence threshold of 80%
    router = QueryRouter(confidence_threshold=0.8)

    print("--- Handling a high-confidence query ---")
    high_confidence_query = "Can you write a python script to list files?"
    response1 = router.handle_query(high_confidence_query)
    print("\nFinal Response to User:\n" + response1)

    print("\n" + "="*40 + "\n")

    print("--- Handling a low-confidence query (fallback) ---")
    low_confidence_query = "help me"
    response2 = router.handle_query(low_confidence_query)
    print("\nFinal Response to User:\n" + response2)