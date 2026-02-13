import torch
from .gradgpt_utils import guess_next_weights

class GradGPT(torch.optim.Optimizer):
    """
    AI-powered optimiser 🚀
    """
    def __init__(self, params, llm="llama3.2:1b", llm_provider="ollama"):
        """
        Create an optimiser instance.
        
        :param params: Arguments to the optimisation problem being solved; must be a `torch.nn.Parameter` or `torch.Tensor` with `grad`.
        :param llm: Name of LLM to invoke for weight updates. Must be compatible with LangChain's `init_chat_model`. Defaults to `llama3.2:1b`.
        :param llm_provider: Name of provider of `llm`. Must be compatible with LangChain's `init_chat_model`. Defaults to `ollama`.
        """
        super().__init__(params, defaults={"llm": llm, "llm_provider": llm_provider})
    
    def step(self):
        for group in self.param_groups:
            for weight in group["params"]:
                llm = group["llm"]
                llm_provider = group["llm_provider"]
                
                with torch.no_grad():
                    weight.copy_(guess_next_weights(weight, llm, llm_provider))