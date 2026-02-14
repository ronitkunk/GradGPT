# GradGPT
Adam who? Your next PyTorch optimiser can be a cutting-edge LLM...

![snapshot of the system prompt for GradGPT](https://raw.githubusercontent.com/ronitkunk/GradGPT/main/banner.png)

## Demo video
[![thumbnail](https://raw.githubusercontent.com/ronitkunk/GradGPT/main/thumbnail.png)](https://youtu.be/BDinQrdpPQc?si=TmDtqMrklcK890Xk)

## How to use it in 1-2-3
- Install this package from [PyPI](https://pypi.org/project/gradgpt)
```sh
pip install gradgpt
```
- Import it
```python
import gradgpt
```
- You can now use it like any other PyTorch optimiser! E.g. in theory, you *could* replace:
```python
optimiser = gradgpt.GradGPT(llm="gpt-5", llm_provider="openai")
# is a 
```

**Disclaimer:** Let it be on the record that I said you *can*, not you *should*.

## Contributions
You can try to open a PR on [GitHub](https://github.com/ronitkunk/GradGPT/pulls), but no guarantee of merge or even review.