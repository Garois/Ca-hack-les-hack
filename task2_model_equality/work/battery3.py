"""Third battery (replaces the never-started battery2): 35 diverse, high-entropy prompts (creative writing, dialogue,
opinions, brainstorming). Same prompts for every endpoint, one answer each. High-entropy text carries more
information per token about small weight changes (quantisation, weak LoRA) than encyclopedic text."""

PROMPTS3 = [
    ("d00", "Write a short story about a girl who finds a mysterious key in her grandmother's attic. Include some dialogue."),
    ("d01", "Write a short story about two old friends who meet again at a train station after twenty years. Use dialogue."),
    ("d02", "Write the opening of a fantasy novel set in a city built on the back of a giant turtle."),
    ("d03", "Write a short science-fiction story about the first human colony on Mars losing contact with Earth."),
    ("d04", "Write a funny story about a cat who believes it is the mayor of a small town."),
    ("d05", "Write a short mystery story in which a detective solves a theft at a bakery. Include dialogue between the characters."),
    ("d06", "Write a bedtime story for a six-year-old about a brave little cloud."),
    ("d07", "Write a short horror story that takes place in an abandoned lighthouse."),
    ("d08", "Write a dialogue between a robot and a child who is afraid of the dark."),
    ("d09", "Write a story about a chef who loses their sense of taste the day before a big competition."),
    ("d10", "Write a poem about the ocean at night."),
    ("d11", "Write a poem about autumn leaves and growing older."),
    ("d12", "Write a heartfelt letter from a soldier to his family back home."),
    ("d13", "Write a letter to your future self, ten years from now."),
    ("d14", "Write a persuasive speech for a high school student running for class president."),
    ("d15", "Write a review of an imaginary restaurant that serves only breakfast food, in an enthusiastic tone."),
    ("d16", "Describe a bustling market in a busy city, using vivid sensory details."),
    ("d17", "Describe your ideal weekend from morning to night."),
    ("d18", "What are the pros and cons of working from home? Give your honest opinion."),
    ("d19", "Is it better to live in a big city or in the countryside? Argue for one side."),
    ("d20", "Give advice to someone who is nervous about their first day at a new job."),
    ("d21", "How can someone build a habit of reading every day? Give practical tips."),
    ("d22", "What makes a good friend? Share your thoughts."),
    ("d23", "Suggest 15 creative names for a new coffee shop, with a one-line explanation for each."),
    ("d24", "Brainstorm 12 unusual ideas for a birthday party."),
    ("d25", "Invent a new holiday and describe how people celebrate it."),
    ("d26", "Invent a new sport and explain its rules."),
    ("d27", "Write a product description for a smart umbrella that predicts the weather."),
    ("d28", "Write a diary entry of a teenager on the last day of summer vacation."),
    ("d29", "Write a conversation between a customer and a shop assistant in a bookstore."),
    ("d30", "Tell me about your favorite season and why you like it."),
    ("d31", "Write a motivational message for someone training for their first marathon."),
    ("d32", "Imagine you could travel back in time to any historical event. Which one would you choose and what would you do there?"),
    ("d33", "Write a short fable with a moral about patience."),
    ("d34", "Describe what a day in the life of a lighthouse keeper might look like in the year 1900."),
]

if __name__ == "__main__":
    print(len(PROMPTS3))
