# It bothers me that ' token' and 'token' are different tokens in standard BPE
# tokenizers! But now I know that the transformer isn't operating on the
# tokens; it's operating on embedding vectors. Maybe the space tokens are close
# to the unspaced tokens in the embedding space? That would make sense and make
# me feel a lot less bad.

import torch
from transformer_lens import HookedTransformer

gpt2_small = HookedTransformer.from_pretrained("gpt2-small")

# In [6]: gpt2_small.to_str_tokens(" token")
# Out[6]: ['<|endoftext|>', ' token']

# In [7]: gpt2_small.to_str_tokens("token")
# Out[7]: ['<|endoftext|>', 'token']

# In [8]: gpt2_small.to_tokens(" token")
# Out[8]: tensor([[50256, 11241]], device='cuda:0')

# In [9]: gpt2_small.to_tokens("token")
# Out[9]: tensor([[50256, 30001]], device='cuda:0')

# In [13]: _token_embedding = gpt2_small.W_E[11241]

# In [14]: token_embedding = gpt2_small.W_E[30001]

# In [17]: from torch.nn import CosineSimilarity

# In [19]: cosine_similarity = CosineSimilarity(dim=0)

# In [20]: cosine_similarity(token_embedding, _token_embedding)
# Out[20]: tensor(0.7406, device='cuda:0', grad_fn=<SumBackward1>)

# Not as high as you'd hope, but maybe it's still pretty high if most vectors
# are approximately orthogonal (because of the curse of dimensionality)? And
# maybe the space means they appear in systematically different contexts, and
# that's what the embedding space is tracking, even if, as a human, I think the
# tokens should "mean the same thing"?

# Let's systematically compare the similarities of spaced-vs.-unspaced tokens
# vs. random tokens. Actually, throw in capitalized-vs.-uncapitalized for good
# measure.

# And actually, furthermore, it would be really interesting if there were a
# consistent vector direction between the space–unspace and cap–uncap pairs
# (reminiscent of the word2vec arithmetic results). Check for that, too?

import random
import statistics
from torch.nn import CosineSimilarity

cosine_similarity = CosineSimilarity(dim=0)

sample_size = 1000


def sample_random_token_pairs():
    return [
        (random.randint(0, 50255), random.randint(0, 50255)) for _ in range(sample_size)
    ]


def sample_random_complemented_token_pairs(candidate_criterion, complement_conversion):
    token_pairs = []
    while len(token_pairs) < sample_size:
        candidate = random.randint(0, 50255)
        candidate_as_string = gpt2_small.to_single_str_token(candidate)
        if candidate_criterion(candidate_as_string):
            complement_as_string = complement_conversion(candidate_as_string)
        else:
            continue

        try:
            complement = gpt2_small.to_single_token(complement_as_string)
        except AssertionError:
            print(
                "complement {!r} to candidate {!r} is not a token; continuing to sample ...".format(
                    complement_as_string,
                    candidate_as_string
                )
            )
            continue

        # Sorting should suffice for consistent order: ' ' comes before ASCII;
        # upper ASCII comes before lower ASCII
        token_pairs.append(tuple(sorted([candidate, complement])))
    return token_pairs


def toggle_leading_space(string):
    if string.startswith(' '):
        return string[1:]
    else:
        return ' ' + string

def toggle_leading_cap(string):
    if string[0].isupper():
        return string[0].lower() + string[1:]
    else:
        return string[0].upper() + string[1:]

def sample_random_space_unspace_token_pairs():
    return sample_random_complemented_token_pairs(lambda s: True, toggle_leading_space)

def sample_random_cap_uncap_token_pairs():
    return sample_random_complemented_token_pairs(lambda s: not s.startswith(' '), toggle_leading_cap)

def compute_token_pair_similarity_statistics(token_pairs):
    similarities = []
    diffs = []
    for t1, t2 in token_pairs:
        e1 = gpt2_small.W_E[t1]
        e2 = gpt2_small.W_E[t2]
        similarity = cosine_similarity(e1, e2)
        similarities.append(similarity.item())
        diffs.append(e2 - e1)
    return statistics.mean(similarities), statistics.stdev(similarities), sum(diffs).norm()


def experiment():
    random_token_pairs = sample_random_token_pairs()
    random_space_unspace_token_pairs = sample_random_space_unspace_token_pairs()
    random_cap_uncap_token_pairs = sample_random_cap_uncap_token_pairs()
    for label, token_pairs in [
        ("random", random_token_pairs),
        ("space–unspace", random_space_unspace_token_pairs),
        ("Cap–uncap", random_cap_uncap_token_pairs),
    ]:
        mean, stdev, mean_diff_norm = compute_token_pair_similarity_statistics(token_pairs)
        print("{} similarities: {} ± {} (norm of avg. diff {})".format(label, mean, stdev, mean_diff_norm))

# random similarities: 0.26880363837629556 ± 0.05365778342462878 (norm of avg. diff 151.92608642578125)
# space–unspace similarities: 0.6557188059091568 ± 0.119279314498032 (norm of avg. diff 648.2448120117188)
# Cap–uncap similarities: 0.7097804792672395 ± 0.20387443294434096 (norm of avg. diff 374.9312744140625)

# TODO: maybe use PCA to determine if there's a consistent direction to the diffs?

if __name__ == "__main__":
    experiment()
