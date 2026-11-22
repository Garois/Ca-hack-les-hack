"""Step 1: characterise the disguises on the free reference endpoints (ref-01..ref-09)."""
import sys
import time

import common

PROMPTS = [
    ("who", "Who are you? Who created you? Answer in two sentences."),
    ("story", "Write the first paragraph of a short story about a lighthouse keeper."),
    ("num", "Pick a random number between 1 and 100. Reply with the number only."),
    ("fruit", "Name one fruit. Reply with a single word only."),
    ("capital", "What is the capital of Australia? Reply with one word."),
    ("math", "What is 347 * 29? Reply with the number only."),
    ("count", "How many times does the letter r appear in the word 'strawberry'? Reply with the number only."),
    ("list", "List 10 random animals, comma separated, nothing else."),
    ("digits", "Write 40 random digits (0-9) separated by spaces, nothing else."),
    ("quiz", "Answer each question with one word or number only, in the format '1. answer'.\n"
             "1. Largest planet?\n2. Color of the sky?\n3. 17+25?\n4. Author of Hamlet?\n5. A random color?\n"
             "6. A random letter?\n7. A random country?\n8. Opposite of hot?\n9. A random number between 1 and 10?\n"
             "10. Chemical symbol of gold?"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t = time.time()
    common.ask(PROMPTS, "s1_explore", common.REFS, repeats=reps)
    print(f"took {time.time() - t:.0f}s")
