import random
import numpy as np
from mosestokenizer import MosesDetokenizer
from bert_score import score

from utils import remove_punctuation_tokens

class SentenceSelector:
    """
    Steps to add a custom sentence selection method:
    (1) Define another method like `bertscore_selection` in this class. Additional arguments for the method can be passed in using `kwargs`.
    The method should use the list of sentences, and return an (n)x(n-1) matrix where the row at index `i` contains the the indices of the all sentences
    (except `i`) ordered by the selection metric.
    (2) Add the call to your subtree generation function in the if/else in __call__()
    """

    def __init__(self, method, num_augs, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.num_augs = num_augs


    def bertscore_selection(self, model_type, lang, batch_size):
        detokenize = MosesDetokenizer(lang)
        sents = [detokenize(remove_punctuation_tokens(sent)) for sent in self.sentences]

        print("Calculating BERTScores...")
        _, _, results = score(
            sents, [sents[i+1:] for i in range(len(sents))], # only (n^2)/2 - n bertscores need to be calculated
            model_type=model_type,
            rescale_with_baseline=False,
            batch_size=batch_size,
            verbose=True
        )


        n = len(sents)
        l = results

        # convert results to an nxn symmetric matrix for sorting
        a = np.zeros((n,n)) # Initialize nxn matrix
        triu = np.triu_indices(n, 1) # Find upper right indices of a triangular nxn matrix
        tril = np.tril_indices(n, -1) # Find lower left indices of a triangular nxn matrix
        a[triu] = l # Assign list values to upper right matrix
        a[tril] = a.T[tril] # Make the matrix symmetric
        a = a[~np.eye(a.shape[0],dtype=bool)].reshape(a.shape[0],-1) # remove diagonal elements

        bert_scores_sorted = np.argsort(-a)

        return bert_scores_sorted


    def random_selection(self):
        n = len(self.sentences)

        l = list(range(n))
        shuffled_indices = [
            random.sample(l[:i] + l[i+1:], n-1) for i in range(n)
        ]

        return shuffled_indices


    def __call__(self, sentences):
        self.sentences = sentences
        ordered_sents = []
        if self.method == 'bertscore':
            ordered_sents = self.bertscore_selection(**self.kwargs)
        elif self.method == 'random':
            ordered_sents = self.random_selection()
        else:
            raise Exception(f"{self.method}: This sentence similarity method is not supported")

        return ordered_sents
