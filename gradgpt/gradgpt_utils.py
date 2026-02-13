import torch
from typing import List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class TensorUpdate(BaseModel):
    values: List[float] = Field(
        ...,
        description="Flattened list of all tensor values in row-major order. Length MUST exactly match the provided tensor size."
    )

initialised_model = None

def guess_next_weights(param: torch.Tensor, llm: str = "llama3.2:1b", llm_provider: str = "ollama") -> torch.Tensor:
    """
    Weight update using LLM.
    """
    global initialised_model

    if param.grad is None:
        raise ValueError("Parameter has no gradient. Make sure to use params of an instance of nn.Module or manually set requires_grad=True!")

    device = param.device
    dtype = param.dtype

    param_flat = param.detach().flatten().cpu().tolist()
    grad_flat = param.grad.detach().flatten().cpu().tolist()

    n = len(param_flat)

    system_prompt = """
You are an expert gradient-based optimiser.

You will be performing a single iteration of a first-order gradient-based optimisation algorithm on the abstract loss function of a machine learning model.

You will be given:
- A tensor of current model parameters (flattened)
- A tensor of the gradient of the loss with respect to said parameters (flattened)

You should, in general:
- aim to change the parameters in the opposite direction of the gradient
- aim to make smaller changes if the gradient is small, and larger changes if the gradient is large

Deliverable:
Return the NEXT parameter tensor values after applying your chosen update rule.

Rules:
- You MUST output exactly N floats in the structured output schema, where N is the tensor size.
- Preserve ordering.
- Do not explain anything.
- Do not include extra fields.
- Populate every element.

You may invent any optimisation strategy you like.
"""

    if initialised_model is None:
        model = init_chat_model(model=llm, model_provider=llm_provider).with_structured_output(TensorUpdate)
        initialised_model = model
    else:
        model = initialised_model

    user_prompt = f"""
Tensor size: {n}

Current parameters (flattened):
{param_flat}

Gradients (flattened):
{grad_flat}

Return the updated parameters.
"""

    result = model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    values = result.values

    if not (len(values) == n):
        print(f"[INFO] LLM attempted to return {len(values)} scalars, {n} are required; retrying.")
        return guess_next_weights(param, llm, llm_provider)

    new_tensor = torch.tensor(values, dtype=dtype).reshape(param.shape).to(device)

    return new_tensor