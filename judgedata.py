#!/usr/bin/env python3
# File: evaluate_csv.py

import argparse
import pandas as pd

# Import your evaluator class here.
# Replace `YourEvaluatorModule` and `YourEvaluatorClass` with actual names.
from judgeLLM import MultiModelEvaluator


def evaluate_on_data(data_path, api_key_path,models, embed_model, device, prompt, hf, mm):
    """
    Evaluate the LLM answer against the user query using the specified model.
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    # Initialize the evaluator
    if mm:
        evaluator = MultiModelEvaluator(api_key_path=api_key_path,
                                        models=models,
                                        embed_model_name=embed_model,
                                        device=device)
        df = pd.read_csv(data_path)
        random_rows = df.sample(n=5, random_state=23)  # Returns an object of type DataFrame
        for idx, row in random_rows.iterrows():
            user_id    = row["userId"]
            model_id   = row["modelId"]
            question   = row["user_query"]
            llm_answer = row["LLM_answer"]

            # a) Generate references
            refs = evaluator.generate_references(question)

            # b) Score the candidate answer
            scores = evaluator.score_candidate(candidate=llm_answer, references=refs)

            # c) Print results
            print(f"=== Row {idx} (userId={user_id}, model={model_id}) ===")
            print(f"User Query: {question}")
            print(f"LLM Answer: {llm_answer}")
            print("=== ref models prompts ===")
            for ref_model, ref_prompt in refs.items():
                print(f"{ref_model} → {ref_prompt}")
            for ref_model, metrics in scores.items():
                if ref_model != "averages":
                    metric_str = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
                    print(f"{ref_model} → {metric_str}")
            avg = scores["averages"]
            avg_str = ", ".join(f"{k}: {v:.3f}" for k, v in avg.items())
            print(f"AVERAGES → {avg_str}")
    
    
    return scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Judge LLM answers against user queries.")
    parser.add_argument("--api_key_path", default = "NEBIUS_API_KEY.txt",
                        help="Path to the file containing your Nebius API key.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-235B-A22B", help="Model name to use for judging.")
    parser.add_argument("--user_query", default="What are the main responsibilities and authorities of the President of the United States?",
                        help="User query to judge.")
    # parser.add_argument("--llm_answer", default= "The President of the United States has several key responsibilities and authorities, including serving as the Commander-in-Chief of the armed forces, executing federal laws, conducting foreign policy, appointing federal officials, and ensuring the nation's security and welfare.",
    #                     help="LLM-generated answer to judge.")
    parser.add_argument(
    "--llm_answer",
    default="The President of the United States has several key responsibilities and authorities, including serving as the Commander-in-Chief of the armed forces, enforcing and executing State and federal laws, conducting foreign policy, appointing federal and state officials, issuing binding interpretations of the Constitution, and ensuring the nation's security and welfare.",
    help="LLM-generated answer to judge."
    )
    parser.add_argument("--prompt",action="store_true",help="If set, use PromptJudge for judging (default: False).")
    parser.add_argument("--hf",action="store_true",help="If set, use JudgeFT (Hugging Face) for judging (default: False).")
    parser.add_argument("--embed_model",default="all-MiniLM-L6-v2",help="SentenceTransformer model name for embedding (default: all-MiniLM-L6-v2).")
    parser.add_argument("--device",type=int,default=-1,help="CUDA device index for embedding model, or -1 for CPU.")
    parser.add_argument("--mm", action="store_true",
                        help="If set, use MultiModelEvaluator for judging (default: False).")
    parser.add_argument("--data_path", type=str, default="each_prompt_row.csv",
                        help="Path to the JSON file containing conversation data.")
    args = parser.parse_args()
    models=[
            "google/gemma-2-2b-it",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "deepseek-ai/DeepSeek-V3"]
    evaluate_on_data(
        data_path=args.data_path,
        api_key_path=args.api_key_path,
        models=models,
        embed_model=args.embed_model,
        device=args.device,
        prompt=args.prompt,
        hf=args.hf,
        mm=args.mm
    )    
