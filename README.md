# GradGPT
Adam who? Why not let a cutting-edge LLM decide your next weight update?

![snapshot of the system prompt for GradGPT](https://raw.githubusercontent.com/ronitkunk/GradGPT/main/banner.png)

### How to use it in 1-2-3
- Install this package from [PyPI](pypi.org/project/gradgpt)
```sh
pip install gradgpt
```
- Import it
```python
import gradgpt
```
- You can now use it like any other PyTorch optimiser! E.g. in theory, you *could* replace:
```
torch.optim.SGD(lr=1e-3)
```
with:
```
gradgpt.GradGPT(llm="gpt-5")
```
**Disclaimer:** Let it be on the record that I said you *can*, not you *should*.