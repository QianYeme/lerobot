# Isolation

No ACT source changes. Separate2/3-input MLP32-32-1, GELU, tanh times bound. Additional load input adds32 first-layer weights. Seeded common hidden and zero output initialization. C1 global frozen, never fine-tuned; arms copied unchanged. C0 is reused globally trained load model, unlike local-only training; asymmetry disclosed.
