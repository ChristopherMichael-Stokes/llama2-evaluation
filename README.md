# LLaMa2 evaluation repo

This repo serves as an experimentation playground for usage and performance evaluation on the LLaMa2 model family.

Currently covers:
- [x] Loading model in various precisions including quantized
- [x] Running standard inference + properly formatted prompt based inference of the chat models 
- [x] Evaluation of NF4 vs int8 performance on a few benchmarks

Future TODOs:

- [ ] QLoRA training
- [ ] Evaluation of 16-bit precisions + the larger LLaMa models
- [ ] Implement batch inference for each of the evaluations
- [ ] Evaluation against other non-llama models + upstream derivitives from the base models