### Data Process
   Generate gold and predict plan pairs:
  
   For prompting results from different VLMs(e.g., Cluade, GPT-4V, Gemini-Pro-1.5), the gold files may be slight different.

   For Gemini-Pro-1.5, we use plan_gold_2.txt

   `python generate_sentence_pair_gemini.py`
   
   For Claude and GPT-4V, we use plan_gold_1.txt

   `python generate_sentence_pair.py`

   It is feasible to use plan_gold_1.txt or plan_gold_2.txt for Claude. We report results of claude with plan_gold_1.txt in the paper.

### Atomatic evaluation
   With the generated plan pairs, run `lcs.py` to calculate least common subsequence score.
   
   `python lcs.py`

