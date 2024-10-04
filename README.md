# ActPlan-1K
This is the code repo for EMNLP 2024 main conference paper: ActPlan-1K: Benchmarking the Procedural Planning Ability of Visual Language Models in Household Activities.

## Counterfactual Activity Definition
   The definition is based on BDDL language. It is an extention of [Behavior100](https://github.com/StanfordVL/bddl/tree/v1.0.1). Our dataset redefines the activity based on the seed activities in Behavior100, utilitizing the [annotation tool](https://behavior.stanford.edu/activity-annotation) and [annotation interface](https://github.com/StanfordVL/behavior-activity-annotator/tree/main?tab=readme-ov-file).

   Details of the annotation steps:
   
   1) For each activity in Behavior100, translate the BDDL description into natural language.

   2) Given each activity, request ChatGPT for specific procedures and situated circumstances which might happend during the process. The prompting contexts are the folder `chatgpt/`

   3) Ground the situated circumstance in igibson enviorment and annotate initial and goal description with [tool](https://stanfordvl.github.io/behavior-activity-annotation/). It will generate a new BDDL case. Or directly modify the normal activity BDDL file to build a counterfactual activity (careful check over it is required as it will later be used to generate scene instances for image collection)

   4) Convert the bddl description into natural language task description, as prompting context.

   The collected counterfactual activity definitions are placed under folder `./bddl/activity-definitions`

## Multimodal Dataset Collection
   Besides the natural language task description from BDDL files, vision information of the enviroments are another key part. To acquire vision information, images covering the main contents of the activity in the environments are collected. Detailed procedures are as follows:

   1. For counterfactual activities, scene instances are first sampled with the activity-definitions in last step, by following the [instructions](https://stanfordvl.github.io/iGibson/sampling.html). The sampled results are in `urdf` file forms.

      For normal activities, we use the predefined activities in [Behavior100](https://github.com/StanfordVL/bddl/tree/v1.0.1). The sampled scene instances can be directly downloaded from [iGibson2 data](https://stanfordvl.github.io/iGibson/dataset.html).

   2. With the sampled counterfactual activity and downloaded normal activity urdf instances, then load them in iGibson2 simulator follow the example in iGibson [sample loader](https://github.com/StanfordVL/iGibson/tree/master/igibson/utils/data_utils/sampling_task/sampling_loader.py). 

   3. Record vedios when touring the house after loading the scene instances. Select images that cover the main contents from the recorded image collections. The sampled images will be used as additional visual input in prompting. 
   
   An example ActPlan-1K instance is under folder `./annotation`. The full dataset including all annotations and sampled urdfs for counterfactual activities will be released soon.

## Auto Evaluation

   With the natural language description and selected image set, we prompt VLMs(e.g., GPT-4V, Claude, Gemini-pro-1.5) to generate procedural plans. The generated plans and gold plans are compared by both human metrics and automatic metrics. We provide two automatic evaluation metrics: longest common subsequence(LCS) and finetuned BLEURT score.

### 1. LCS
   Details are in folder `./auto_lcs`

### 2. Finetuned BLEURT score
   Details are in folder `./bleu-cls`

