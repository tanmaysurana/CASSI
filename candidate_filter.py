from mosestokenizer import MosesDetokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import remove_punctuation_tokens


class CandidateFilter:

    def __init__(
            self,
            dataset, 
            candidates,
            lang,
            scoring_lib,
            lm,
            batch_size,
            device,
            no_len_norm,
            no_jscore,
            jscore_k,
            prescored,
            **kwargs
        ):
        self.dataset = dataset
        self.candidates = pd.DataFrame(candidates)
        self.lang = lang
        self.scoring_lib = scoring_lib
        self.lm = lm
        self.batch_size = batch_size
        self.device = device
        self.no_len_norm = no_len_norm
        self.no_jscore = no_jscore
        self.jscore_k = jscore_k
        self.prescored = prescored
        self.kwargs = kwargs


    def jaccard(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union


    def mlm_score_mlmscoring(self, sents):
        from mlm.scorers import MLMScorer, MLMScorerPT
        from mlm.models import get_pretrained, SUPPORTED_MLMS
        import mxnet as mx

        ctxs = [mx.gpu(0)] if self.device == 'cuda' else [mx.cpu()]
        model, vocab, tokenizer = get_pretrained(ctxs, self.lm)
        lm_scorer = None
        if self.lm in SUPPORTED_MLMS:
            lm_scorer = MLMScorer(model, vocab, tokenizer, ctxs)
        else:
            lm_scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
        
        lm_scores = lm_scorer.score_sentences(sents, split_size=self.batch_size)

        return lm_scores
    

    def mlm_score_minicons(self, sents, pll_metric):
        from minicons import scorer
        from torch.utils.data import DataLoader

        batches = DataLoader(sents, batch_size=self.batch_size)
        lm_scorer = scorer.MaskedLMScorer(self.lm, self.device)
        lm_scores = []
        for batch in tqdm(batches):
            batch_scores = lm_scorer.sequence_score(
                list(batch), 
                reduction = lambda x: x.sum(0).item(), 
                PLL_metric=pll_metric
            )
            lm_scores.extend(batch_scores)
        return lm_scores


    def candidate_filter(self):
        cand_df = self.candidates.groupby(['Sent1_Id', 'Sent2_Id'], sort=False).agg(list).reset_index()

        augs = []
        for _, row in cand_df.iterrows():
            scored_cands = list(zip(
                row['lm_score'], 
                row['tokens'], 
                row['tags']
            ))

            if not self.no_jscore:
                scored_cands.sort(reverse=True)
                top_k_cands = scored_cands[:self.jscore_k]

                jscores = []
                for cand in top_k_cands:
                    sent1_l = [x.lower() for x in dataset[row['Sent1_Id']]['tokens']]
                    sent2_l = [x.lower() for x in dataset[row['Sent2_Id']]['tokens']]
                    cand_l = [x.lower() for x in cand[1]]
                    jaccard1 = self.jaccard(sent1_l, cand_l)
                    jaccard2 = self.jaccard(sent2_l, cand_l)
                    jscores.append(np.sqrt(jaccard1 * jaccard2))

                aug_i = int(np.argmax(jscores))
                augs.append({
                    'Sent1_Id': row['Sent1_Id'],
                    'Sent2_Id': row['Sent2_Id'],
                    'tokens': top_k_cands[aug_i][1],
                    'tags': top_k_cands[aug_i][2],
                })
            else:
                aug = max(scored_cands)
                augs.append({
                    'Sent1_Id': row['Sent1_Id'],
                    'Sent2_Id': row['Sent2_Id'],
                    'tokens': aug[1],
                    'tags': aug[2],
                })
        
        return augs


    def __call__(self):
        if not self.prescored:
            detokenize = MosesDetokenizer(self.lang)
            cands = [cand for cand in self.candidates.loc[:, 'tokens']]
            detok_cands = np.array([detokenize(cand) for cand in cands])

            lm_scores = []
            if self.scoring_lib == 'mlmscoring':
                lm_scores = self.mlm_score_mlmscoring(detok_cands)
            elif self.scoring_lib == 'minicons':
                lm_scores = self.mlm_score_minicons(detok_cands, **self.kwargs)

            if not self.no_len_norm:
                cand_lens = np.array([len(list(remove_punctuation_tokens(cand))) for cand in cands])
                lm_scores = np.array(lm_scores) / cand_lens
            
            self.candidates['lm_score'] = lm_scores
        
        augs = self.candidate_filter()

        return augs, self.candidates.to_dict(orient='records')
        
                

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import pickle

    from utils import get_dataset, write_aug_files
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--candidates_file', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--others_file', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--encoding',
        type=str,
        required=False,
        default='utf-8'
    )
    parser.add_argument(
        '--output_base_name', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--lang', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--num_augs', 
        type=int, 
        required=True
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda'
    )
    parser.add_argument(
        '--prescored',
        action='store_true'
    )
    parser.add_argument(
        '--scoring_lib',
        type=str,
        required=False,
        default='mlmscoring',
        choices=['mlmscoring', 'minicons']
    )
    parser.add_argument(
        '--lm',
        type=str,
        required=False,
        default='bert-base-multi-cased'
    )
    parser.add_argument(
        '--scoring_batch_size',
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        '--no_len_norm',
        action='store_true'
    )
    parser.add_argument(
        '--no_jscore',
        action='store_true'
    )
    parser.add_argument(
        '--jscore_k',
        type=int,
        required=False,
        default=5
    )
    parser.add_argument(
        '--pll_metric',
        type=str,
        required=False,
        default='within_word_l2r',
        choices=['within_word_l2r', 'original']
    )

    args = parser.parse_args()

    dataset = get_dataset(args.input_file, args.encoding)
    with open(args.candidates_file, "rb") as f_cands:
        candidates = pickle.load(f_cands, encoding=args.encoding)

    sent_augs, scored_candidates = CandidateFilter(
        dataset=dataset,
        candidates=candidates,
        lang=args.lang,
        scoring_lib=args.scoring_lib,
        lm=args.lm,
        batch_size=args.scoring_batch_size,
        device=args.device,
        pll_metric=args.pll_metric,
        no_len_norm=args.no_len_norm,
        no_jscore=args.no_jscore,
        jscore_k=args.jscore_k,
        prescored=args.prescored
    )()

    # write scored candidates
    with open(Path('intermediate_aug_files') / f"candidates_{args.output_base_name}.pkl", "wb") as f_cands:
        pickle.dump(scored_candidates, f_cands)

    # write aug files
    with open(args.others_file, "rb") as f_others:
        other_augs = pickle.load(f_others, encoding=args.encoding)
    write_aug_files(
        base_name=args.output_base_name,
        num_augs=args.num_augs,
        dataset=dataset,
        sent_augs=sent_augs,
        other_augs=other_augs,
        encoding=args.encoding
    )


