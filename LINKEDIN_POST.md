# LinkedIn Post — qwen-to-gemma-math

---

Distilling reasoning from a frontier model into a small on-device model is a multi-stage engineering problem.

You need a high-quality teacher, a prompt format that produces structured traces rather than collapsed answers, a filtering pass to remove degenerate outputs, a fine-tuning loop that handles architecture constraints, and an evaluation harness that tells you whether the student actually learned reasoning or just memorised templates.

Most teams prototype one piece at a time. The full pipeline rarely gets built.

NEO built the entire thing autonomously — from trace generation to HuggingFace publish — for `google/gemma-4-E2B-it` using `qwen/qwen3.6-plus` as the teacher on GSM8K math problems.

200 chain-of-thought traces generated via OpenRouter API. 193 passed quality filtering (96.5%). NEO discovered mid-run that PEFT/LoRA is incompatible with Gemma-4's `Gemma4ClippableLinear` layers and switched to full fine-tuning without intervention. 5 training epochs. Loss dropped 27.7%.

The distilled student model produces genuine multi-step arithmetic — not answer templates. 35% of test problems solved correctly on numeric evaluation. The model is live at huggingface.co/daksh-neo/qwen-to-gemma-math.

No code was written by hand. No architecture decisions were made by a human. NEO handled the full research-to-deployment loop.

Try NEO → heyneo.so

---

*#AI #MachineLearning #KnowledgeDistillation #LLM #AutonomousAI #NEO #GSM8K #ChainOfThought #OpenSource*
