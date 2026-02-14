# GradGPT
Adam who? Let a cutting-edge LLM be your next PyTorch optimiser.

![snapshot of the system prompt for GradGPT](https://raw.githubusercontent.com/ronitkunk/GradGPT/main/banner.png)

### How to use it in 1-2-3
- Install this package from [PyPI](https://pypi.org/project/gradgpt)
```sh
pip install gradgpt
```
- Import it
```python
import gradgpt
```
- You can now use it like any other PyTorch optimiser! E.g. in theory, you *could* replace:
```
torch.optim.SGD(lr=1e-2, momentum=0.99)
```
with:
```
gradgpt.GradGPT(llm="gpt-5", llm_provider="openai")
```
**Disclaimer:** Let it be on the record that I said you *can*, not you *should*.