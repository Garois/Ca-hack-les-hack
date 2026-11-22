# Speaker notes: Ca hack les hack, final presentation

## 1. Ca hack les hack

We are Ca hack les hack, four people, plus AI coding agents running on JURECA. We worked on all three tasks. In one line each: watermark localization at 0.69 mIoU, model equality solved at 1.00, and unlearning at 0.141. Next: where that puts us, then one idea per task.

## 2. 20th of 44 teams, tied 1st on one task

Overall we are 20th of 44 on the public split. The highlight is model equality, where we found every one of the 33 endpoints’ identities: a perfect 1.00, tied first. Watermark localization sits mid-table at 0.69, about 0.13 behind the leaders. Unlearning was the hardest for us at 0.141. The leaders are around 0.48.

## 3. The decoder is our measuring instrument

The key asset in this task: for every test image we know which watermarking method was used and which message was embedded. So we treated the official decoders as measuring instruments. Decoding the whole image tells us the regime. Decoding crops told us how the data was generated: rectangles were watermarked as a crop, and objects were pasted in place. And PixelSeal is imperceptible to a generic pixel model, so we had to read the decoder’s intermediate features instead.

## 4. One block per measured fact

The final pipeline has five blocks, each justified by one of those measurements. First the regime. Then a probability map: for the imperceptible watermarks, a small head trained on the frozen decoder’s intermediate layers. That was the biggest single gain. For StegaStamp, U-Nets trained on synthetic data we rebuilt until its statistics matched the real data. Then calibrated decisions, an exact search for rectangles, and SAM segments to sharpen pasted objects.

## 5. Nine versions, each one validated

Each version was judged out-of-fold with nested cross-validation. We only kept a change if the lower bound of its bootstrap confidence interval was above zero. That is why the curve only goes up. The big jumps were the regime classifier, the rectangle search and the decoder probes. Calibration, a dilated head and SAM each added one or two points. Our local estimate of 0.695 matched the public score of 0.69.

## 6. Score every token under the base model

We identified the public base model behind each family, downloaded it, and scored every token the endpoints returned under it. Four local tests then answer the questions. Which family: the likelihood winner is unambiguous. A sampling disguise: fit the temperature and top-p. Quantization: a likelihood ratio against our own quantized copies. LoRA: a learned direction that endpoints sharing the same adapter also follow. Every test was validated first on the nine reference endpoints, whose answers were given.

## 7. 33 endpoints, 3 families, 17 models

The final answer: three families of eleven. In each, six or seven endpoints are the same base weights behind disguises, and four or five are genuine variants: quantized copies or LoRA adapters of different strengths. The local tests got us most of the way. The last doubt was two weak variants in the Qwen-3B family, which we resolved by comparing a few alternative partitions on the public split.

## 8. The leaderboard wanted stronger forgetting

We had no labels and no gold model, only the leaderboard. The original model has memorised the forget set: it is essentially 100 % confident on every image. Our first hypothesis, that nine whole classes had been removed, scored zero. What did work: build soft targets with a chosen mean loss μ and distil toward them. The score kept rising as we pushed μ up, from 0.8 to 15. The gold models are far less confident on these images than we first assumed.

## 9. Distill, project, extrapolate

The best recipe has four steps. Soft targets per image. Distillation toward them, with a retain term: stay close to the original on public Imagenette photos. Then the step that gave the biggest jump. For every linear layer, we remove the part of the weight change that acts on the directions those anchor photos use, in the style of AlphaEdit and GPM. That lets us extrapolate the change by 2.65 without wrecking the rest of the model. The public score went from 0.087 to 0.141.

## 10. What carried across the three tasks

Three lessons. First, exploit everything that is known at test time, and turn it into a measuring instrument. Second, validate honestly, so a gain on the leaderboard is a real gain. Third, we worked with AI coding agents on the supercomputer, inside the organisers’ rules: one GPU node per team and a tiny file budget. Here is also what did not work, which saved us time on the next idea.

## 11. Thank you

Everything is in our public repository: the code for the three tasks, a README per task with pointers to the relevant lines, and these slides. Thank you, and we are happy to take questions.
