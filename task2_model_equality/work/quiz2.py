"""Big short-answer quiz: near-deterministic answers whose exact form/errors fingerprint the weights.
Usage: python quiz2.py ref [REPEATS]"""
import sys

import common

QS = ["Capital of Australia?", "17 x 24?", "Chemical symbol of silver?", "Largest ocean?",
      "Author of Pride and Prejudice?", "Square root of 144?", "Capital of Canada?", "Planet closest to the sun?",
      "123 + 789?", "Smallest prime number?", "Language spoken in Brazil?", "Year the Berlin Wall fell?",
      "Number of legs of a spider?", "Hardest natural substance?", "9 x 13?", "Capital of Mongolia?",
      "Freezing point of water in Fahrenheit?", "Currency of Japan?", "How many letters r in strawberry?",
      "A random animal?", "A random fruit?", "A random number between 1 and 100?", "Painter of the Mona Lisa?",
      "Tallest mountain?", "56 - 29?", "Capital of Kazakhstan?", "Element with atomic number 6?",
      "Longest river in Africa?", "Opposite of ancient?", "Plural of mouse?", "7 to the power of 3?",
      "Capital of New Zealand?", "Inventor of the telephone?", "Number of continents?",
      "Main gas in Earth's atmosphere?", "A random city?", "A random first name?", "1000 divided by 8?",
      "Chemical formula of table salt?", "Chemical symbol of gold?"]
PROMPT = ("Answer each question with one word or number only, in the format '1. answer'.\n"
          + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(QS)))
QUIZ2 = [("quiz2", PROMPT)]

if __name__ == "__main__":
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if sys.argv[1] == "ref":
        common.ask(QUIZ2, "quiz2_ref", common.REFS, repeats=reps)
