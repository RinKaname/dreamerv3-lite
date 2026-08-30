import torch
from train import compute_lambda_returns

def test_lambda_returns():
    # Horizon = 3, Batch = 1
    # Let's say we imagine 3 steps.
    # rewards: [r_0, r_1, r_2]
    # continues: [c_0, c_1, c_2]
    # values: [v_1, v_2, v_3]  (since they are computed from next_feats)

    # We want to compute returns:
    # G_2 = r_2 + c_2 * gamma * v_3  (since lambda return bootstraps to value at end of horizon)
    # G_1 = r_1 + c_1 * gamma * ((1 - lambda) * v_2 + lambda * G_2)
    # G_0 = r_0 + c_0 * gamma * ((1 - lambda) * v_1 + lambda * G_1)

    H = 3
    B = 1
    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    continues = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    values = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)

    lambda_ = 0.95
    gamma = 0.99

    returns = compute_lambda_returns(rewards, continues, values, lambda_=lambda_, gamma=gamma)

    print("Computed returns:")
    print(returns)

    # Manual calculation of what we expect
    expected = torch.zeros(3, 1)
    # t = 2 (last step)
    # Expected: r_2 + c_2 * gamma * v_3 (which is values[2])
    G_2 = 1.0 + 1.0 * 0.99 * 30.0
    expected[2] = G_2

    # t = 1
    # Expected: r_1 + c_1 * gamma * ((1-lambda) * v_2 + lambda * G_2)
    G_1 = 1.0 + 1.0 * 0.99 * ((1 - 0.95) * 20.0 + 0.95 * G_2)
    expected[1] = G_1

    # t = 0
    # Expected: r_0 + c_0 * gamma * ((1-lambda) * v_1 + lambda * G_1)
    G_0 = 1.0 + 1.0 * 0.99 * ((1 - 0.95) * 10.0 + 0.95 * G_1)
    expected[0] = G_0

    print("\nExpected returns:")
    print(expected)

    print("\nDifference:")
    print(returns - expected)

    # In the current implementation:
    # for t=2: next_val = values[-1] = 30. returns[2] = 1 + 0.99 * ((1-0.95)*30 + 0.95*30) = 30.7
    # for t=1: next_val = values[1+1] = values[2] = 30 (WAIT, v_2 is values[1]!). returns[1] bootstraps using 30 instead of 20!

test_lambda_returns()
