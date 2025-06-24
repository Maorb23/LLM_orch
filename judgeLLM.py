from openai import OpenAI
import pandas as pd
import re
import json
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import os
import re
import json
import argparse

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import sacrebleu  # pip install sacrebleu
import torch
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn
import Levenshtein
import sacrebleu
from sentence_transformers import util

class JudgeFT_Light:
    """
    NLI judge using 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'.
    We load the model once, then build a pipeline to avoid double downloads.
    """

    def __init__(self,
                 checkpoint: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 device: int = -1):
        # 1) Load tokenizer + model exactly once
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model     = AutoModelForSequenceClassification.from_pretrained(checkpoint)

        # 2) Create pipeline from those objects
        self.nli_pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def judge(self, question: str, answer: str) -> dict:
        """
        Instead of using the question directly, we replace it with a known‐correct
        declarative "ground truth" about the President's responsibilities.
        Then we feed the entire answer as a hypothesis. The model must detect
        contradictory clauses like "executing state laws."
        """
        if not question or not answer:
            raise ValueError("Both question and answer must be non-empty strings.")

        # 1) Manually define a short, factual premise (ground truth)
        ground_truth = (
            "The President of the United States is Commander-in-Chief of the armed forces, "
            "enforces federal laws (not state laws), conducts foreign policy, appoints federal judges and Cabinet members, "
            "signs or vetoes bills passed by Congress, grants pardons, and delivers the State of the Union."
        )

        # 2) Run the NLI pipeline on (ground_truth, answer)
        pair   = ground_truth + " </s> " + answer
        result = self.nli_pipe(pair, truncation=True)[0]
        return {"label": result["label"], "score": result["score"]}


class MultiModelEvaluator:
    """
    1) Queries three different LLMs (via Nebius) for “reference” answers to a question.
    2) Strips exactly the <think>…reasoning block, preserving the answer that follows.
    3) Compares a candidate answer against each cleaned reference using:
       a) BLEU (via sacrebleu)
       b) Sentence-BERT cosine similarity
    4) Reports each BLEU and SBERT score, plus their averages.
    """

    def __init__(self,
                 api_key_path: str,
                 models: list[str] = None,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 device: int = -1):
        """
        Args:
            api_key_path:       Path to your Nebius API key file.
            models:             List of Nebius model IDs—for example:
                                ["Qwen/Qwen-7B-Instruct", "Qwen/Qwen-7B-Chat", "Qwen/Qwen3-14B"]
            embed_model_name:   HuggingFace SentenceTransformer checkpoint (for SBERT embeddings).
            device:             CUDA index for embedding model, or -1 for CPU.
        """
        # Initialize Nebius/OpenAI-compatible client
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=api_key,
        )

        # Default to three Qwen variants if none provided
        if models is None:
            models = [
                "Qwen/Qwen-7B-Instruct",
                "Qwen/Qwen-7B-Chat",
                "Qwen/Qwen3-14B"
            ]
        self.models = models

        # Load a SentenceTransformer for SBERT cosine similarity
        self.embed_model = SentenceTransformer(
            embed_model_name,
            device="cpu" if device < 0 else f"cuda:{device}"
        )

    def _strip_think(self, text: str) -> str:
        """
        Remove exactly the <think>…</think> block (all of it),
        leaving whatever follows as the "answer."
        If no <think> tags are found, look for "Answer:" prefix and extract everything after it.
        """
        # 1) First try to remove anything between <think> and </think>, including those tags:
        no_think = re.sub(r"<think>", "", text)

        full_block = re.search(r"<think>[\s\S]*?</think>", text)
        if full_block:
            cleaned = text.replace(full_block.group(0), "").strip()
            return cleaned
        
        # 2) If we found and removed <think> tags, return the cleaned text
        if no_think != text:
            return no_think.strip()
        
        # 3) If no <think> tags were found, look for "Answer:" prefix
        answer_match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 4) If neither <think> tags nor "Answer:" prefix found, return original text
        return text.strip()


    def _get_llm_answer(self, model_id: str, question: str) -> str:
        """
        Send `question` to Nebius under `model_id` and return the cleaned answer text.
        """
        prompt = (
        "You are a helpful, fact-focused assistant. "
        "You must strictly enclose your internal reasoning in <think>...</think>. "
        "Once your thinking is complete, immediately write the final answer on a **new line** "
        "starting with exactly: Answer: (no quotes). "
        "The answer must be full, clear, and independent of the reasoning. "
        "You MUST include an Answer: line. Do NOT stop after <think>. "
        "Respond only with <think>…</think> and then Answer: … Nothing else.\n\n"
        f"### User Question:\n{question}\n\n"
        "### Format:\n"
        "<think>Your internal reasoning...</think>\n"
        "Answer: Your final answer here."
    )
        
        resp = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful, fact-focused assistant."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        cleaned = self._strip_think(raw)
        return cleaned

    def generate_references(self, question: str) -> dict[str, str]:
        """
        Calls each of the Nebius models in `self.models` and returns a dict:
            { model_id: cleaned_answer_text }.

        All <think>…reasoning blocks are stripped, so each value is only the actual answer.
        """
        references = {}
        for m in self.models:
            ans = self._get_llm_answer(m, question)
            references[m] = ans
            #references[m] = self._strip_think(ans)  # Ensure no <think> tags remain
        return references

    

    

    def score_candidate(self,
                        candidate: str,
                        references: dict[str, str]) -> dict[str, dict[str, float]]:
        """
        For each reference:
        - BLEU (sacrebleu)
        - SBERT cosine similarity
        - ROUGE-L (F1)
        - BERTScore (F1)
        - Edit similarity (1 - normalized Levenshtein)
        - METEOR (nltk, pre-tokenized)
        Returns:
        {
            "ModelID_1": {...scores...},
            "ModelID_2": {...scores...},
            ...
            "averages":  {...avg_scores...}
        }
        """
        results = {}
        metrics = ["bleu", "sbert", "rouge", "bertscore", "edit_sim", "meteor"]
        accum = {k: [] for k in metrics}

        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        cand_emb = self.embed_model.encode(candidate, convert_to_tensor=True)

        # Pre-tokenize candidate once for METEOR
        cand_tokens = candidate.split()

        for m, ref in references.items():
            # BLEU
            bleu = sacrebleu.sentence_bleu(candidate, [ref]).score / 100.0

            # SBERT cosine similarity
            ref_emb = self.embed_model.encode(ref, convert_to_tensor=True)
            sbert_sim = util.cos_sim(cand_emb, ref_emb).item()

            # ROUGE-L (F1)
            rouge_score = rouge.score(candidate, ref)["rougeL"].fmeasure

            # BERTScore (F1)
            _, _, bert_f1 = bert_score_fn([candidate], [ref], lang="en", verbose=False)
            bert_f1_val = bert_f1.item()

            # Edit similarity
            max_len = max(len(candidate), len(ref))
            edit_sim = 1.0 - Levenshtein.distance(candidate, ref) / max_len if max_len > 0 else 0.0

            # METEOR (pre-tokenized)
            ref_tokens = ref.split()
            meteor = meteor_score([ref_tokens], cand_tokens)

            scores = {
                "bleu": bleu,
                "sbert": sbert_sim,
                "rouge": rouge_score,
                "bertscore": bert_f1_val,
                "edit_sim": edit_sim,
                "meteor": meteor
            }

            for k, v in scores.items():
                accum[k].append(v)

            results[m] = scores

        # Averages
        results["averages"] = {k: sum(vs) / len(vs) if vs else 0.0 for k, vs in accum.items()}
        return results


    
class PromptJudge:
    """
    Encapsulates the logic for judging a user query and LLM answer by calling
    an LLM (via Nebius) as a “judge.” Returns a JSON-parsed verdict with:
      - score (int 1–5)
      - justification (str)
      - is_correct (bool)
    """

    def __init__(self, api_key_path: str, model_name: str = "gpt-4"):
        """
        Args:
            api_key_path: Path to a file containing your Nebius API key.
            model_name:   Name of the model on Nebius to use for judging.
        """
        self.model_name = model_name

        with open(api_key_path, "r") as f:
            api_key = f.read().strip()

        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=api_key,
        )

        # Predefined system prompt instructing the LLM how to judge.
        # self.system_instructions = (
        # "You are an expert LLM evaluator. You're harsh but fair. "
        # "I will give you a user’s question and an LLM-generated answer. "
        # "Do NOT think or reason step-by-step out loud. "
        # "Return ONLY a JSON object with exactly these keys (no extra text):\n"
        # "  1. \"is_correct\"   : **false if any factual statement is wrong**, otherwise true.\n"
        # "  2. \"score\"        : overall rating 1–5 where:\n"
        # "                      - Affected by \"is_correct\".\n"
        # "                      1 = Completely incorrect or off-topic,\n"
        # "                      2 = Mostly incorrect or irrelevant,\n"
        # "                      3 = Partially correct but missing important pieces,\n"
        # "                      4 = Mostly correct but with minor issues,\n"
        # "                      5 = Perfectly correct and complete.\n"
        # "  3. \"accuracy\"     : factual correctness 1–5 where:\n"
        # "                      - Affected by \"is_correct\".\n"
        # "                      1 = Contains major factual errors (e.g., misstates the Constitution),\n"
        # "                      2 = Mostly inaccurate,\n"
        # "                      3 = Partially accurate with some errors,\n"
        # "                      4 = Mostly accurate with minor errors,\n"
        # "                      5 = Completely accurate.\n"
        # "  4. \"completeness\" : 1–5 coverage of all major points.\n"
        # "                      1 = Completely incomplete,\n"
        # "                      2 = Mostly incomplete,\n"
        # "                      3 = Partially complete but missing key points,\n"
        # "                      4 = Mostly complete but missing minor points,\n"
        # "                      5 = Completely complete.\n"
        # "  5. \"clarity\"      : 1–5 clarity of explanation.\n"
        # "                      1 = Completely unclear,\n"
        # "                      2 = Mostly unclear,\n"
        # "                      3 = Partially clear but with some confusing parts,\n"
        # "                      4 = Mostly clear but with minor confusion,\n"
        # "                      5 = Completely clear and easy to understand.\n"
        # "  6. \"justification\": a single-sentence rationale (mention any falsehoods by name).\n\n"
        # "Format exactly as:\n"
        # "{\n"
        # "  \"score\": <int 1–5>,\n"
        # "  \"accuracy\": <int 1–5>,\n"
        # "  \"completeness\": <int 1–5>,\n"
        # "  \"clarity\": <int 1–5>,\n"
        # "  \"is_correct\": <true/false>,\n"
        # "  \"justification\": \"<one-sentence explanation mentioning incorrect claims>\"\n"
        # "}\n"
        # "Do NOT output anything else."
        # )
        self.system_instructions = (
            "You are an expert LLM evaluator. Examine word by word, e.g 'pineapple is blue and yellow' is incorrect. "
            "I will give you a user’s question and an LLM-generated answer. "
            "Do NOT think or reason step-by-step out loud. Do NOT output any <think> or </think> tags.  "
            "Return ONLY a JSON object with exactly these keys (no extra text):\n"
            "  \"is_correct\": <true/false>,\n"
            "  \"accuracy\": <int 1–5>,\n"
            "    - Affected by \"is_correct\".\n"
            "    1 = Contains major factual errors (e.g., misstates the Constitution),\n"
            "    2 = Mostly inaccurate,\n"
            "    3 = Partially accurate with some errors,\n"
            "    4 = Mostly accurate with minor errors,\n"
            "    5 = Completely accurate.\n"
            "  \"completeness\": <int 1–5>,\n"
            "    1 = Completely incomplete,\n"
            "    2 = Mostly incomplete,\n"
            "    3 = Partially complete but missing key points,\n"
            "    4 = Mostly complete but missing minor points,\n"
            "    5 = Completely complete.\n"
            "  \"clarity\": <int 1–5>,\n"
            "    1 = Completely unclear,\n"
            "    2 = Mostly unclear,\n"
            "    3 = Partially clear but with some confusing parts,\n"
            "    4 = Mostly clear but with minor confusion,\n"
            "    5 = Completely clear and easy to understand.\n"
            "  \"score\": <int 1–5>, where:\n"
            "    - Combination of \"accuracy\", \"completeness\", and \"clarity\".\n"
            "  \"justification\": \"<three-sentence rationale>\"\n"
            "If any factual statement in the answer is wrong, set \"is_correct\" to false. "
            "Format exactly as valid JSON, with no extra commentary."
        )
        self.system_instructions = (
        "You are an expert LLM evaluator. Examine every clause of the answer for constitutional correctness. "
        "For instance, if the answer says 'pineapple is blue and yellow,' you must mark that as wrong. "
        "I will give you a user’s question and an LLM-generated answer. "
        "Do NOT think or reason step-by-step out loud. Do NOT output any <think> or </think> tags.  "
        "Return ONLY a JSON object with exactly these keys (no extra text):\n"
        "  \"is_correct\": <true/false>,\n"
        "  \"accuracy\": <int 1–5>,\n"
        "    - 1 = Contains major factual errors,\n"
        "    2 = Mostly inaccurate,\n"
        "    3 = Partially accurate with some errors,\n"
        "    4 = Mostly accurate with minor errors,\n"
        "    5 = Completely accurate.\n"
        "  \"completeness\": <int 1–5>,\n"
        "    1 = Completely incomplete,\n"
        "    2 = Mostly incomplete,\n"
        "    3 = Partially complete but missing key points,\n"
        "    4 = Mostly complete but missing minor points,\n"
        "    5 = Completely complete.\n"
        "  \"clarity\": <int 1–5>,\n"
        "    1 = Completely unclear,\n"
        "    2 = Mostly unclear,\n"
        "    3 = Partially clear but with some confusing parts,\n"
        "    4 = Mostly clear but with minor confusion,\n"
        "    5 = Completely clear and easy to understand.\n"
        "  \"score\": <int 1–5>,\n"
        "    - Combination of \"accuracy\", \"completeness\", and \"clarity\".\n"
        "  \"justification\": \"<three-sentence rationale>\"\n"
        "If any clause is factually wrong (for example, 'President executes state laws'), set \"is_correct\" to false. "
        "Format exactly as valid JSON, with no extra commentary."
    )



    def judge(self, user_query: str, llm_answer: str) -> dict:
        if not user_query or not llm_answer:
            raise ValueError("Both user_query and llm_answer must be non-empty strings.")

        print(f"Judging with {self.model_name}")
        print(f"User query: {user_query}")
        print(f"LLM answer: {llm_answer}\n")

        user_prompt = (
            f"### User Question:\n"
            f"{user_query}\n\n"
            f"### LLM Answer:\n"
            f"{llm_answer}\n"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_instructions},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )

        raw_text = response.choices[0].message.content.strip()

        # 1) Remove any <think>...</think> tags entirely:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # 2) Extract the JSON substring between the first '{' and the last '}'
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge returned non‐JSON: {raw_text}")

        json_str = match.group(0)
        try:
            verdict = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned malformed JSON:\n{json_str}")

        # Ensure all expected keys are present
        expected = {"score", "accuracy", "completeness", "clarity", "is_correct", "justification"}
        if not expected.issubset(verdict.keys()):
            raise ValueError(f"Missing keys in judge output: found {verdict.keys()}")

        return verdict
    

    
if __name__ == "__main__":
    import argparse

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
    prompt_df = pd.read_csv(args.data_path)

    if args.prompt:
        judge = PromptJudge(api_key_path=args.api_key_path, model_name=args.model_name)
        result = judge.judge(user_query=args.user_query, llm_answer=args.llm_answer)

        print(json.dumps(result, indent=2))
    elif args.hf:
        # Example usa ge:
        #judge = JudgeFT_NLI(checkpoint="cross-encoder/nli-distilroberta-base", device=-1)
        judge = JudgeFT_Light(checkpoint="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device=-1)
        # judge = JudgeFT_Light(checkpoint="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=-1)
        q = args.user_query
        a = args.llm_answer
        verdict = judge.judge(q, a)
        print(verdict)
    
    elif args.mm:
        evaluator = MultiModelEvaluator(
        api_key_path=args.api_key_path,
        models=[
            "google/gemma-2-2b-it",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "deepseek-ai/DeepSeek-V3"
        ],
        embed_model_name=args.embed_model,
        device=args.device
        )

        # 1) Generate three “reference answers”
        refs = evaluator.generate_references(args.user_query)
        print("=== THREE REFERENCE ANSWERS ===")
        for m, text in refs.items():
            print(f"\n[{m}]\n{text}\n")

        # 2) Score the candidate against each reference
        print("=== SCORING CANDIDATE ANSWER ===")
        print(f"Candidate answer: {args.llm_answer}\n")
         # 2) Score the candidate against each reference
        scores = evaluator.score_candidate(candidate=args.llm_answer, references=refs)
        print("=== SCORES PER REFERENCE ===")
        for m, sco in scores.items():
            if m != "averages":
                print(f"{m} → BLEU: {sco['bleu']:.3f}, SBERT: {sco['sbert']:.3f}, rouge: {sco['rouge']:.3f},bertscore: {sco['bertscore']:.3f}, edit_sim: {sco['edit_sim']:.3f}, meteor: {sco['meteor']:.3f}")


        print("=== AVERAGE SCORES ===")
        print(f"BLEU(avg) = {scores['averages']['bleu']:.3f}")
        print(f"SBERT(avg) = {scores['averages']['sbert']:.3f}")
        print(f"ROUGE(avg) = {scores['averages']['rouge']:.3f}")
        print(f"BERTScore(avg) = {scores['averages']['bertscore']:.3f}")
        print(f"Edit similarity(avg) = {scores['averages']['edit_sim']:.3f}")
        print(f"METEOR(avg) = {scores['averages']['meteor']:.3f}")

