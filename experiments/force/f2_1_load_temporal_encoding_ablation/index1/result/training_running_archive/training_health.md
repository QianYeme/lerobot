# 早期训练健康

- A0: step=100, loss=5.308899402618408, grad_norm(before clip)=179.10736083984375, elapsed=11.931949412450194s, exit=None
- A1: step=42000, loss=0.07051869481801987, grad_norm(before clip)=8.175970077514648, elapsed=3724.246579926461s, exit=None
- A2: step=42100, loss=0.0830695852637291, grad_norm(before clip)=6.7869343757629395, elapsed=3721.9943198636174s, exit=None

这是启动快照，不是100k全程健康结论。grad_norm为clip前范数，实际每步clip10；非有限loss/梯度会立即终止并记录非零exit。尚无正式Dev checkpoint成绩。
