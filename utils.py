import pandas as pd
import csv
from pathlib import Path
import unicodedata
import pickle
import codecs


def get_dataset(input_file, encoding):
    df = pd.read_csv(
        input_file, 
        sep='\t', 
        names=['Word', 'Tag'], 
        skip_blank_lines=False, 
        quoting=csv.QUOTE_NONE, 
        encoding=encoding
    )

    sent_id = 0
    sent_ids = []
    for i, row in df.iterrows():
        if row.isnull().values.all():
            sent_id += 1
            df.drop(i, inplace=True)
            continue
        sent_ids.append(sent_id)

    df['Sent_Id'] = sent_ids
    df = df.groupby(['Sent_Id'], sort=False).agg(list).reset_index()
    df.set_index('Sent_Id', inplace=True)

    dataset = [{ 
        'tokens': row['Word'], 
        'tags': row['Tag'] 
    } for _, row in df.iterrows()]

    return dataset


def write_aug_files(base_name, num_augs, dataset, sent_augs, other_augs, encoding):
    # write sents exclusively
    with open(Path('intermediate_aug_files') / f"sents_{base_name}.pkl", "wb") as f_sents:
        pickle.dump(sent_augs, f_sents)

    # write augmentation file with dataset + sents + others
    for n in range(1, num_augs+1):
        with codecs.open(Path('aug_files') / f"{base_name}_{n}.txt", "w", encoding) as f_augs:
            for d in dataset:
                for i, tok in enumerate(d['tokens']):
                    f_augs.write(f"{tok}\t{d['tags'][i]}\n")
                f_augs.write("\n")
            for j in range(0, len(sent_augs), num_augs):
                for s in sent_augs[j:j+n]:
                    for i, tok in enumerate(s['tokens']):
                        f_augs.write(f"{tok}\t{s['tags'][i]}\n")
                    f_augs.write("\n")
            for j in range(0, len(other_augs), num_augs):
                for o in other_augs[j:j+n]:
                    for i, tok in enumerate(o['tokens']):
                        f_augs.write(f"{tok}\t{o['tags'][i]}\n")
                    f_augs.write("\n")

    return None


def remove_punctuation_tokens(tokens):
    return list(filter(
                lambda t: not (len(t) == 1 and unicodedata.category(t).startswith('P')), 
                tokens
            ))
