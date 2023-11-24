import random
import spacy
from spacy.tokens import Doc
from spacy.glossary import GLOSSARY
from tqdm import tqdm
import re
from sentence_selector import SentenceSelector



class CandidateGenerator:
    """
    Steps to add a dependency parser:
    (1) Define another method like `get_spacy_subtrees` in this class. Additional arguments for the parser can be passed in using `kwargs`.
        The method should use `self.dataset`, and return indices of subtrees of all sentences in a list in the following format:
        ->  [{'subj_subtrees': [(subtree_start, subtree_end), ...], 'obj_subtrees': [(subtree_start, subtree_end), ...]}]
        where each entry in the list is a collection of 'subj_subtrees'+'obj_subtrees' of sentence
    (2) Add the call to your subtree generation function in the if/else in __call__()
    """

    def __init__(
            self,
            dataset,
            parser,
            num_augs,
            sentence_selector,
            sent_filter_len,
            **kwargs
        ):
        self.dataset = dataset
        self.parser = parser
        self.num_augs = num_augs
        self.sentence_selector = sentence_selector
        self.sent_filter_len = sent_filter_len
        self.kwargs = kwargs

        self.tag_dict = {}
        for sent in self.dataset: # build a dictionary of entities by type for random replacement
            tag_type = None
            entity = None
            for i, tag in enumerate(sent['tags']):
                if tag.startswith('B'):
                    tag_type = tag[2:]
                    if tag_type not in self.tag_dict:
                        self.tag_dict[tag_type] = []
                    entity = [sent['tokens'][i]]
                elif tag.startswith('I'):
                    if entity is None:
                        tag_type = tag[2:]
                        entity = [sent['tokens'][i]]
                    else:
                        entity.append(sent['tokens'][i])
                    if (i+1) == len(sent['tags']):
                        self.tag_dict[tag_type].append(entity)
                        tag_type = None
                        entity = None
                elif tag == 'O':
                    if entity is not None and tag_type is not None:
                        self.tag_dict[tag_type].append(entity)
                        tag_type = None
                        entity = None


    def get_spacy_subtrees(self):
        subj_tags = []
        obj_tags = []

        for k,v in GLOSSARY.items(): # collect all tags that refer to 'subject', 'object', and 'complement' from spacy
            if 'subject' in v.replace(',', ' ').split():
                subj_tags.append(k)
            if ('complement' in v.replace(',', ' ').split()) or ('object' in v.replace(',', ' ').split()):
                obj_tags.append(k)

        spacy.prefer_gpu()
        nlp = spacy.load(self.parser)

        print("Parsing Sentences...")
        docs = [nlp(Doc(nlp.vocab, d['tokens'])) for d in tqdm(self.dataset)]

        print("Generating Subtrees...")
        subtrees = []
        for doc in tqdm(docs):
            subtrees.append({
                'subj_subtrees': [],
                'obj_subtrees': []
            })
            for _, token in enumerate(doc):
                dep = token.dep_.split(':')[0]
                if dep in subj_tags:
                    subtree = [(t.i, t.text) for t in list(token.subtree)]
                    subtree.sort() # sort because sometimes spacy returns subtree tokens out of place
                    subtrees[-1]['subj_subtrees'].append((subtree[0][0], subtree[-1][0] + 1))
                if dep in obj_tags:
                    subtree = [(t.i, t.text) for t in list(token.subtree)]
                    subtree.sort() # sort because sometimes spacy returns subtree tokens out of place
                    subtrees[-1]['obj_subtrees'].append((subtree[0][0], subtree[-1][0] + 1))

        return subtrees


    def fix_iob_tags(self, tags):
        for j, tag in enumerate(tags): # fix NER tags according to IOB scheme
                if tag != 'O':
                    if j == 0 and tag.startswith('I'): # |I -> |B
                        tags[j] = 'B' + tag[1:]
                    elif j > 0:
                        if tags[j-1] == 'O' and tag.startswith('I'): # OI -> OB
                            tags[j] = 'B' + tag[1:]
                        elif tags[j-1].startswith('B') and tag == tags[j-1]: # BB -> BI
                            tags[j] = 'I' + tag[1:]
                        elif tags[j-1].startswith('I') and tag[1:] == tags[j-1][1:]: # IB -> II
                            tags[j] = 'I' + tag[1:]
        return tags


    def get_subtree_swaps(self, ix_1, ix_2):
        sent1 = self.dataset[ix_1]
        sent2 = self.dataset[ix_2]
        subtrees_1 = self.subtrees[ix_1]
        subtrees_2 = self.subtrees[ix_2]
        cands = []
        for p1 in subtrees_1['subj_subtrees']:
            for p2 in subtrees_2['subj_subtrees']:
                cands.append({
                    'tokens': sent1['tokens'][:p1[0]] + sent2['tokens'][p2[0]:p2[1]] + sent1['tokens'][p1[1]:],
                    'tags': self.fix_iob_tags(sent1['tags'][:p1[0]] + sent2['tags'][p2[0]:p2[1]] + sent1['tags'][p1[1]:]),
                })

        for p1 in subtrees_1['obj_subtrees']:
            for p2 in subtrees_2['obj_subtrees']:
                cands.append({
                    'tokens': sent1['tokens'][:p1[0]] + sent2['tokens'][p2[0]:p2[1]] + sent1['tokens'][p1[1]:],
                    'tags': self.fix_iob_tags(sent1['tags'][:p1[0]] + sent2['tags'][p2[0]:p2[1]] + sent1['tags'][p1[1]:]),
                })

        for p1 in subtrees_2['subj_subtrees']:
            for p2 in subtrees_1['subj_subtrees']:
                cands.append({
                    'tokens': sent2['tokens'][:p1[0]] + sent1['tokens'][p2[0]:p2[1]] + sent2['tokens'][p1[1]:],
                    'tags': self.fix_iob_tags(sent2['tags'][:p1[0]] + sent1['tags'][p2[0]:p2[1]] + sent2['tags'][p1[1]:]),
                })

        for p1 in subtrees_2['obj_subtrees']:
            for p2 in subtrees_1['obj_subtrees']:
                cands.append({
                    'tokens': sent2['tokens'][:p1[0]] + sent1['tokens'][p2[0]:p2[1]] + sent2['tokens'][p1[1]:],
                    'tags': self.fix_iob_tags(sent2['tags'][:p1[0]] + sent1['tags'][p2[0]:p2[1]] + sent2['tags'][p1[1]:]),
                })

        return cands


    def get_other_aug(self, ix):
        tokens = []
        ner_tags = []
        for i, tag in enumerate(self.dataset[ix]['tags']):
            if tag.startswith("B"):
                tag_type = tag[2:]
                rep_tokens = random.choice(self.tag_dict[tag_type])
                tokens.extend(rep_tokens)
                ner_tags.extend([f'B-{tag_type}'] + [f'I-{tag_type}']*(len(rep_tokens)-1))
            elif tag == 'O':
                tokens.append(self.dataset[ix]['tokens'][i])
                ner_tags.append('O')

        return { 'tokens': tokens, 'tags': ner_tags }


    def get_candidates(self):
        # separate out sents and others (with og indices)
        sents = [] # sentences where a subject/object/complement are detected are called 'sents'
        others = [] # sentences where a subject/object/complement are NOT detected are called 'others'
        for i, s in enumerate(self.subtrees):
            if (len(s['subj_subtrees']) == 0 and len(s['obj_subtrees']) == 0) or (len(self.dataset[i]['tokens']) <= self.sent_filter_len):
                others.append(i)
            else:
                sents.append(i)

        assert 0 < self.num_augs < len(sents), "Number of augmentations must be smaller than the number of sentences where augmentation can be applied"

        # get ranked sentences
        ordered_sents = self.sentence_selector(sentences=[self.dataset[i]['tokens'] for i in sents])
        selected_sents = []
        # perform sent replacement
        candidates = []
        for i, selected_ix in enumerate(ordered_sents):
            selected_sents.append([])
            for j, ix in enumerate(selected_ix):
                if j < i and i in selected_sents[j]: continue
                cands = self.get_subtree_swaps(sents[i], sents[ix])
                _cands = []
                for cand in cands:
                    if len(cand['tokens']) <= self.sent_filter_len: continue
                    _cands.append({
                        'Sent1_Id': sents[i],
                        'Sent2_Id': sents[ix],
                        'tokens': cand['tokens'],
                        'tags': cand['tags']
                    })
                if len(_cands) > 0:
                    candidates.extend(_cands)
                    selected_sents[i].append(ix)
                    if len(selected_sents[i]) == self.num_augs: break

        assert len({len(s) for s in selected_sents}) == 1, "Number of augmentations is too high to generate equal number of augmentations for all sentences"

        # perform other replacement
        other_augs = []
        for other in others:
            for _ in range(self.num_augs):
                aug = self.get_other_aug(other)
                other_augs.append({
                    'Sent_Id': other,
                    'tokens': aug['tokens'],
                    'tags': aug['tags']
                })

        return candidates, other_augs


    def __call__(self):
        if re.match("^.*_(core|dep)_.*$", self.parser):
            self.subtrees = self.get_spacy_subtrees()
        else:
            raise Exception(f"{self.parser}: This dependency parser is not supported")

        return self.get_candidates()



if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path

    from utils import get_dataset


    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument(
        '--input_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output_base_name',
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


    # Sentence Selector specific arguments
    parser.add_argument(
        '--sentence_selector',
        type=str,
        required=False,
        default='bertscore'
    )
    parser.add_argument(
        '--selector_model',
        type=str,
        required=False,
        default='xlm-roberta-base'
    )
    parser.add_argument(
        '--selector_batch_size',
        type=int,
        required=False,
        default=512
    )


    # Candidate Generator specific arguments
    parser.add_argument(
        '--parser',
        type=str,
        required=True
    )
    parser.add_argument(
        '--sent_filter_len',
        type=int,
        required=False,
        default=5
    )


    args = parser.parse_args()

    # load dataset
    dataset = get_dataset(args.input_file, args.encoding)

    # 1. set up sentence selector
    sent_selector = SentenceSelector(
        method=args.sentence_selector,
        num_augs=args.num_augs,
        model_type=args.selector_model,
        lang=args.lang,
        batch_size=args.selector_batch_size
    )

    # 2. get candidates
    candidates, other_augs = CandidateGenerator(
        dataset=dataset,
        parser=args.parser,
        num_augs=args.num_augs,
        sentence_selector=sent_selector,
        sent_filter_len=args.sent_filter_len
    )()

    # write candidates and other_augs to files
    with open(Path('intermediate_aug_files') / f"candidates_{args.output_base_name}.pkl", "wb") as f_cands, \
        open(Path('intermediate_aug_files') / f"others_{args.output_base_name}.pkl", "wb") as f_others:
        pickle.dump(candidates, f_cands)
        pickle.dump(other_augs, f_others)

    print('Stored Candidates.')