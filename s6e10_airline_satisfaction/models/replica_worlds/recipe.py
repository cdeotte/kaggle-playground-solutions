"""Settings of the two replica worlds, shared by train_generator.py, sample_world.py and make_world.py."""

LLM = 'gpt2-medium'
BATCH = 8
EPOCHS = 40           # length of the linear learning-rate schedule (5e-5 -> 0)
MILESTONE = 32        # the worlds are sampled from the weights at the end of this epoch
RATING_JITTER = 0.05  # P(-1) = P(+1) for every service rating in the generator's training copy

# label_flip: (P(False -> True), P(True -> False)) in the generator's training copy of data.csv.
# prior: P('satisfaction is True,') in the sampling prompt = positive rate of that noisy copy (rounded).
# posthoc_flip: rate of symmetric label flips added to the assembled world.
WORLDS = {
    'M32_1M': dict(label_flip=(0.07, 0.07), prior=0.444179, posthoc_flip=0.009),
    'MB32_1M': dict(label_flip=(0.085, 0.090), prior=0.443455, posthoc_flip=0.0),
}
