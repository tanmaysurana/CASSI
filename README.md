# CASSI: Contextual and Semantic Structure-based Interpolation

This is the code for our paper "CASSI: Contextual and Semantic Structure-based Interpolation Augmentation for Low-Resource NER" accepted in EMNLP 2023 (Findings).

## Dependencies

```bash
# SpaCy
pip install spacy[cuda111] # replace with your cuda version/cpu version
python -m spacy download en_core_web_trf # spacy pipeline to be used (https://spacy.io/usage)

# BERTScore
pip install git+https://github.com/tanmaysurana/bert_score.git

# minicons (if using the minicons library)
pip install minicons==0.2.17

# MLM Scoring (if using the mlm-scoring library)
pip install mxnet-cu102mkl # Replace w/ your CUDA version; mxnet-mkl if CPU only.
pip install git+https://github.com/awslabs/mlm-scoring.git

# Note: mlm-scoring and spacy might require separate environments due to conflicting dependencies
```

## Usage
```bash
# Step 1: Candidate Generation

python candidate_generator.py \
    --input_file <input_file_path> \ # dataset path
    --output_base_name <output_base_name> \ # base name used by generated files
    --lang en \
    --num_augs 10 \
    --parser en_core_web_trf # spacy pipeline or custom parser name
    # --device cuda \
    # --sentence_selector bertscore \ "random" for random selection
    # --selector_model xlm-roberta-base \
    # --selector_batch_size 512 \
    # --encoding utf-8 # encoding for the text


# Step 2: Candidate Filtering

python candidate_filter.py \
    --input_file <input_file_path> \ # same as above
    --output_base_name <output_base_name> \ 
    --candidates_file <candidate_file_path> \ # by default stored in intermediate_aug_files 
    --others_file <others_file_path> \ # by default stored in intermediate_aug_files
    --lang en \
    --num_augs 10 \ # must be same as above
    # --device cuda
    # --lm bert-base-multi-cased \
    # --scoring_batch_size 512 \
    # --scoring_lib mlmscoring \ # or minicons, we use mlmscoring in our paper
    # --encoding utf-8
    # --prescored # to run a different filtering method on scored candidates
    # --no_jscore # to select sentences using best lm score

```
_Note_: Custom dependency parsers and sentence selection methods can be added by following the instructions in `candidate_generator.py` and `sentence_selector.py` respectively. 

The above files accept the dataset in the two-column format (separated by tabs):
```
BEIJING B-LOC
1996-08-22      O

Consultations   O
should  O
be      O
held    O
to      O
set     O
the     O
time    O
and     O
format  O
of      O
the     O
talks   O
```
<!-- ## Citation
Please cite our paper if you found this repository useful.

```

``` -->
