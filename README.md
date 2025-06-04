# tryaii

### Folders:

We have empty dev and prod folders for placeholders, scripts on main.

### Scripts:
- extract_json.py - Extracts csv from json. creates for **each query** a new row.
- judgeLLM.py - our classes with main for testing on generated queries and answers. Right now focuesed only on **Non-code** queries and answers. Adding code suppport might be hard to do good but can use same logic for an above averag judge initially.
- judgedata.py - runnig the classes on 5 random queries and answers from a generated csv with extract_json.py.

### Classes in judgeLLM.py:
- **PromptJudge** - Using prompt to judge the answers.
- **MultiModelEvaluator** - Using **multiple LLM models** to evaluate the answers. We get "labels" (answers) from the model and compute metrics compared to the generated answer the user got. The current metrics are:
 - *bleu-* measures the similarity between the generated answer and the model's answer, using a score computed by:
 ![alt text](bleu.png)
 - *rouge-* measures the overlap between the generated answer and the model's answer, computing the overlap of n-grams.
 - *bertscore-* measures the similarity between the generated answer and the model's answer, using a score computed by bert embeddings.
 - *sbert-* measures the similarity between the generated answer and the model's answer, using a score computed by cosine similarity of sentence embeddings using 'all-MiniLM-L6-v2' model.
 - edit_sim - measures the similarity between the generated answer and the model's answer, using a score computed by Levenshtein distance:
 $$
 lev(a, b) = \begin{cases}
  |a| + |b| & \text{if } a = \emptyset \text{ or } b = \emptyset \\
  lev(a[1:], b[1:]) & \text{if } a[0] = b[0] \\
  1 + min(lev(a, b[1:]), lev(a[1:], b), lev(a[1:], b[1:])) & \text{otherwise}
    \end{cases}
 $$
- Meteor - measures the similarity between the generated answer and the model's answer, using a score computed by:
 $$
 Meteor(a, b) = \frac{1}{\sum_{i=1}^{n} \frac{1}{\text{len}(a_i)}} \sum_{i=1}^{n} \text{len}(a_i) \cdot \text{len}(b_i)
 $$
 where $a_i$ and $b_i$ are the n-grams of the generated answer and the model's answer, respectively.
 - **JudgeFT_Light** - Using finetuned huggingface model on fact checks and Q&A. Tried using multiple models, but the results were not good. Maybe a differnt approach or models trained on different data than mnli, nli and fever.

### Usage:
- nebius_api_key- Go to nebius ai studio and generate an api key and plug it in a Nebius_api_key.txt file in the main directory.
- dev and prod folders - add a dev and prod folder in the main directory, they are empty for now.
- run extract_json.py - this will create a csv file with the queries and answers from the json file.
- test judgeLLM.py and judgedata.py.
