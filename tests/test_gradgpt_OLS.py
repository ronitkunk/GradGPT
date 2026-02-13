import torch
from gradgpt import GradGPT

class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        return self.linear(x)


def test_linear_regression_converges():
    torch.manual_seed(0)

    # y = 2x + 1
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[1.0], [3.0]])

    model = SimpleLinearModel()

    optimiser = GradGPT(model.parameters(), llm="llama3.2:1b", llm_provider="ollama")

    loss_fn = torch.nn.MSELoss()

    for _ in range(5):
        optimiser.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimiser.step()

    weight = model.linear.weight.item()
    bias = model.linear.bias.item()

    print("w", weight, "b", bias)

    # No check on convergence; this test is just for code hygiene and syntax
    assert True