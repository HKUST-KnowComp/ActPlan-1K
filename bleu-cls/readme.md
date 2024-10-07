### Data Preprocess
   We have processed the annotations and predictions from VLMs. Under folder `./data`, activities have been randomly split into train/valid/test splits with 60%/20%/20% splits. Details are in `./data/activities.txt`.

   The entire dataset for the activities are in `./data/finetuning_data.jsonl`. For each instance, 

   `pos_gpt`: if the gpt prompting result is correct.

   `neg_gpt`: if the gpt prompting result is wrong.

   `gold_gpt`: gold plan annotation from plan_gold_1.txt

   `pos_gemini`: if the gemini-pro prompting result is correct.

   `neg_gemini`: if the gemini-pro prompting result is wrong.

   `gold_gemini`: gold plan annotation from plan_gold_2.txt

   Positive pairs are formed by pos_gpt, pos_gemini, gold_gpt and gold_gemini

   Negative pairs are formed by one sentence from neg_gpt and neg_gemini (or random shuffled senetences), and one sentence from pos_gpt, pos_gemini, gold_gpt and gold_gemini.


### Training and Evaluation

   Finetuning with BLEURT-base model:
   ```
   CUDA_VISIBLE_DEVICES=0 python finetune_large.py
   ```

   Finetuning with BLEURT model:
   ```
   CUDA_VISIBLE_DEVICES=0 python finetune.py
   ```
