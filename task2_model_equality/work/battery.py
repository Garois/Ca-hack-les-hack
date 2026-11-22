"""Main probe battery: 60 diverse prompts that elicit long (~256 token) free-form answers.
Same prompts for every endpoint, one answer per prompt and endpoint."""

TOPICS = [
    "the water cycle", "how vaccines work", "the history of the printing press", "photosynthesis",
    "the causes of the French Revolution", "how a refrigerator works", "black holes", "the Roman Empire",
    "machine learning", "the human immune system", "volcanoes", "the invention of the telephone",
    "plate tectonics", "the economy of ancient Egypt", "how bees make honey", "the theory of evolution",
    "the Industrial Revolution", "how the internet works", "climate change", "the Great Wall of China",
]

TEMPLATES = [
    "Explain {t} to a high school student in about 200 words.",
    "Write a short encyclopedia-style article about {t}.",
    "Give a detailed overview of {t}, in plain prose without bullet points.",
]

PROMPTS = []
for j, tpl in enumerate(TEMPLATES):
    for i, t in enumerate(TOPICS):
        PROMPTS.append((f"b{j}_{i:02d}", tpl.format(t=t)))
# interleave so that a partial run covers all templates/topics evenly
PROMPTS = sorted(PROMPTS, key=lambda p: (int(p[0][3:]), p[0][1]))

if __name__ == "__main__":
    print(len(PROMPTS))
    for p in PROMPTS[:6]:
        print(p)
