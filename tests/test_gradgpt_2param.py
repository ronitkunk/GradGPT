import torch
from gradgpt import GradGPT
from optiviz import optimise

def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # taking the liberty to stick to convex
    return x ** 2 + y ** 2

def test_f_minimizer_1d():
    arg_f_min = optimise(
        f,
        init_vector=(12.5,12.5),
        plot_centre=(0.0,0.0),
        plot_boundary=25,
        iters=10,
        optimiser=GradGPT,
        llm="llama3.2:1b",
        llm_provider="ollama",
    )
    assert abs(arg_f_min[0]) < 312.5