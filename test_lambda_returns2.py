import torch
import math

def compute_lambda_returns(rewards, continues, next_values, lambda_=0.95, gamma=0.99):
    """Computes generalized lambda returns for Actor-Critic."""
    H, B = rewards.shape[:2]
    returns = torch.zeros_like(next_values)

    # Bootstrap from the last value
    last_val = next_values[-1]

    for t in reversed(range(H)):
        returns[t] = rewards[t] + continues[t] * gamma * ((1 - lambda_) * next_values[t] + lambda_ * last_val)
        last_val = returns[t]
    return returns

def test():
    H = 3
    B = 1
    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    continues = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    next_values = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)

    lambda_ = 0.95
    gamma = 0.99

    returns = compute_lambda_returns(rewards, continues, next_values, lambda_=lambda_, gamma=gamma)

    expected = torch.zeros(3, 1)

    # G_2 = r_2 + c_2 * gamma * ((1 - lambda) * v_2 + lambda * v_2)
    # wait: the formula is G_t = r_t + c_t * gamma * ((1-lambda) * v_{t} + lambda * G_{t+1})
    # For t=2, G_{t+1} is the bootstrapped value past the horizon.
    # What should that be? Usually it's next_values[-1].
    # So for t=2: G_2 = r_2 + c_2 * gamma * ((1-lambda)*next_values[2] + lambda*next_values[-1])
    # Which is exactly r_2 + c_2 * gamma * next_values[2]

    v_last = next_values[-1]
    for t in reversed(range(H)):
        expected[t] = rewards[t] + continues[t] * gamma * ((1 - lambda_) * next_values[t] + lambda_ * v_last)
        v_last = expected[t]

    print(returns - expected)

test()
