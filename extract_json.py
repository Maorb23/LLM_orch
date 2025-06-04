import json
import pandas as pd
import argparse


class ConversationPreprocessor:
    """
    Loads a JSON array of conversation documents and converts each user prompt +
    its next assistant response into a row of a pandas DataFrame.

    Each row contains:
      - userId
      - query_number       (running count per userId)
      - title
      - modelId
      - user_query
      - user_timestamp
      - LLM_answer
      - answer_timestamp
      - PromptTokens
      - CompletionTokens
      - LatencyMS
      - Cost
    """

    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to a JSON file containing a top-level array of documents.
        """
        self.json_path = json_path
        self.docs = self._load_json()
        # Tracks how many prompts we've seen per userId
        self._per_user_count = {}

    def _load_json(self) -> list[dict]:
        """Reads the JSON file and returns the list of documents."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_rows(self) -> list[dict]:
        """
        Iterates over all documents and yields one dict per user-prompt+assistant-response pair.
        """
        rows: list[dict] = []

        for doc in self.docs:
            user_id = doc.get("userId")
            title   = doc.get("title")
            top_mid = doc.get("modelId")

            # Initialize counter for this user if necessary
            if user_id not in self._per_user_count:
                self._per_user_count[user_id] = 0

            messages = doc.get("messages", [])
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg.get("role", "").lower() == "user":
                    # Capture the user prompt text + timestamp
                    user_text = msg.get("content", "").strip()
                    user_ts   = None
                    if isinstance(msg.get("timestamp"), dict):
                        user_ts = msg["timestamp"].get("$date")

                    # Look ahead for the very next assistant message
                    assistant_text    = None
                    answer_ts         = None
                    prompt_tokens     = None
                    completion_tokens = None
                    latency_ms        = None
                    cost_val          = None

                    for j in range(i + 1, len(messages)):
                        candidate = messages[j]
                        if candidate.get("role", "").lower() == "assistant":
                            assistant_text = candidate.get("content", "").strip()

                            # extract assistant timestamp
                            if isinstance(candidate.get("timestamp"), dict):
                                answer_ts = candidate["timestamp"].get("$date")

                            # extract assistant metadata
                            meta = candidate.get("metadata", {})
                            prompt_tokens     = meta.get("promptTokens")
                            completion_tokens = meta.get("completionTokens")
                            latency_ms        = meta.get("latencyMs")
                            cost_val          = meta.get("cost")
                            break

                    # Only emit a row if we found a matching assistant response
                    if assistant_text is not None:
                        self._per_user_count[user_id] += 1
                        query_no = self._per_user_count[user_id]

                        rows.append({
                            "userId":           user_id,
                            "query_number":     query_no,
                            "title":            title,
                            "modelId":          top_mid,
                            "user_query":       user_text,
                            "user_timestamp":   user_ts,
                            "LLM_answer":       assistant_text,
                            "answer_timestamp": answer_ts,
                            "PromptTokens":     prompt_tokens,
                            "CompletionTokens": completion_tokens,
                            "LatencyMS":        latency_ms,
                            "Cost":             cost_val
                        })

                    # Move on to the next message after this user entry
                    i += 1
                else:
                    i += 1

        return rows

    def to_dataframe(self) -> pd.DataFrame:
        """
        Processes the loaded documents and returns a pandas DataFrame
        with one row per user prompt + assistant response.
        """
        rows = self._extract_rows()
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser(description="Preprocess conversation JSON into a DataFrame.")
    parser.add_argument("--json_path", type=str,default="prod/tryaii_production_chats.json",
                         help="Path to the JSON file containing conversation data.")
    args = parser.parse_args()
    preprocessor = ConversationPreprocessor(json_path=args.json_path)
    df = preprocessor.to_dataframe()
    print(df.head())

    # Optionally save to CSV or Excel:
    df.to_csv("each_prompt_row.csv", index=False)
    # df.to_excel("each_prompt_row.xlsx", index=False)
