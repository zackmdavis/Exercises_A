# Tailcalled challenged me to answer what goes wrong when you try to fit a
# naïve Bayes model to data that isn't conditionally independent. (Subsequent
# discussion narrowed this down to a situation where some of the variables are
# being generated by the same latent, but others aren't, but your model doesn't
# know about this.)

# So, let's simulate some data like that, fit it with sci-kit learn, and try to
# see if there's anything detectably "wrong"?

from statistics import NormalDist

def generate_samples(category, n):
    data = []

    match category:
        case "♀":
            μ = 0.3
        case "♂":
            μ = -0.3

    for _ in range(n):
        # There's a Gaussian latent variable x, and the x_i for a given
        # person/data-thingy are that plus some noise.
        latent_x = NormalDist(μ, 1).samples(1)[0]

        x1 = latent_x + NormalDist(0., 0.1).samples(1)[0]
        x2 = latent_x + NormalDist(0., 0.2).samples(1)[0]
        x3 = latent_x + NormalDist(0., 0.3).samples(1)[0]
        x4 = latent_x + NormalDist(0., 0.4).samples(1)[0]

        # But the y_i are separate Gaussians.
        y1 = NormalDist(μ + 0.1, 1).samples(1)[0]
        y2 = NormalDist(μ + 0.2, 1).samples(1)[0]
        y3 = NormalDist(μ + 0.3, 1).samples(1)[0]
        y4 = NormalDist(μ + 0.4, 1).samples(1)[0]

        data.append([x1, x2, x3, x4, y1, y2, y3, y4])

    return data

import numpy as np
from sklearn.naive_bayes import GaussianNB

f_samples = generate_samples("♀", 50)
m_samples = generate_samples("♂", 50)
f_labels = ["F"] * 50
m_labels = ["M"] * 50

samples = np.array(f_samples + m_samples)
labels = np.array(f_labels + m_labels)

model = GaussianNB()
model.fit(samples, labels)