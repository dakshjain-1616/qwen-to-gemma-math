# LinkedIn Post — qwen-to-gemma-math

---

I gave an AI agent a single instruction:

**"Distill reasoning from a powerful API model into a small on-device model. Figure it out."**

No code written by hand. No human in the loop. Here's what NEO built — start to finish.

---

**The task:** Transfer mathematical reasoning from `qwen/qwen3.6-plus` (a frontier API model) into `google/gemma-4-E2B-it` (a 5B on-device model) using chain-of-thought distillation on GSM8K.

**What NEO did autonomously:**

→ Designed the full distillation pipeline from scratch
→ Called the OpenRouter API to generate 200 step-by-step math reasoning traces using qwen3.6-plus as teacher
→ Filtered traces for quality — 193/200 passed (96.5% valid, avg 48 words each)
→ Discovered that PEFT/LoRA is incompatible with Gemma-4's architecture and switched to full fine-tuning automatically
→ Trained the student model over 5 epochs — loss dropped 27.7% (27.44 → 19.57)
→ Evaluated on 20 GSM8K test problems, wrote the evaluation script, parsed results
→ Published the model to HuggingFace with a full model card

**Results:**
- 10% strict GSM8K accuracy (2/20 exact match)
- **35% numeric accuracy** — the model consistently produces correct answers, just with trailing punctuation the evaluator penalises
- BLEU 0.329 vs teacher traces (range 0.171–0.629)
- Student outputs genuine multi-step algebra — not templates, not guesses

The student went from zero to writing things like:

*"Let T be the total. Sold 1/3T at green, 2 at red, 1/2(2/3T - 2) at orange. T - 1/3T - 2 - (1/3T - 1) = 5 → 1/3T = 6 → T = 18."* ✓

**The model is live:** huggingface.co/daksh-neo/qwen-to-gemma-math

NEO didn't just write code — it debugged architecture blockers, adapted the approach mid-run, managed disk constraints, filtered bad data, and shipped a working distilled model.

This is what autonomous AI development looks like in 2025.

Try NEO → heyneo.so

---

*#AI #MachineLearning #LLM #KnowledgeDistillation #AutonomousAI #NEO #OpenSource #GSM8K #Gemma #ChainOfThought*
