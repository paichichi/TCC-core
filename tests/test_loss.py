import torch

from hralign.losses import human_robot_contrastive_loss


def naive_loss(human, frozen_robot, adapted_robot, temperature):
    terms_h2r = []
    terms_r2h = []
    for index in range(human.shape[0]):
        positive = torch.exp(
            human[index].dot(adapted_robot[index]) / temperature
        )
        baseline = torch.exp(
            human[index].dot(frozen_robot[index]) / temperature
        )
        unpaired_h2r = sum(
            torch.exp(human[index].dot(adapted_robot[j]) / temperature)
            for j in range(human.shape[0])
            if j != index
        )
        unpaired_r2h = sum(
            torch.exp(adapted_robot[index].dot(human[j]) / temperature)
            for j in range(human.shape[0])
            if j != index
        )
        terms_h2r.append(
            -torch.log(positive / (positive + baseline + unpaired_h2r))
        )
        terms_r2h.append(
            -torch.log(positive / (positive + baseline + unpaired_r2h))
        )
    return 0.5 * (
        torch.stack(terms_h2r).mean() + torch.stack(terms_r2h).mean()
    )


def test_vectorized_loss_matches_paper_equation():
    torch.manual_seed(2)
    human = torch.randn(4, 7, dtype=torch.float64)
    frozen_robot = torch.randn(4, 7, dtype=torch.float64)
    adapted_robot = torch.randn(4, 7, dtype=torch.float64)
    temperature = 0.7

    actual, _ = human_robot_contrastive_loss(
        human, frozen_robot, adapted_robot, temperature
    )
    expected = naive_loss(
        human, frozen_robot, adapted_robot, temperature
    )
    torch.testing.assert_close(actual, expected)


def test_aligned_features_outperform_shuffled_features():
    torch.manual_seed(3)
    human = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    frozen_robot = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    aligned_loss, aligned_metrics = human_robot_contrastive_loss(
        human, frozen_robot, human.clone(), temperature=0.1
    )
    shuffled_loss, _ = human_robot_contrastive_loss(
        human, frozen_robot, human.roll(1, dims=0), temperature=0.1
    )
    assert aligned_loss < shuffled_loss
    assert aligned_metrics["h2r_top1"] == 1
    assert aligned_metrics["r2h_top1"] == 1


def test_single_pair_still_compares_against_frozen_baseline():
    human = torch.tensor([[1.0, 0.0]])
    frozen_robot = torch.tensor([[0.0, 1.0]])
    adapted_robot = human.clone().requires_grad_()
    loss, metrics = human_robot_contrastive_loss(
        human, frozen_robot, adapted_robot, temperature=1.0
    )
    expected = -torch.log(torch.exp(torch.tensor(1.0)) / (torch.e + 1))
    torch.testing.assert_close(loss, expected)
    assert metrics["negative_similarity"] == 0
    loss.backward()
    assert adapted_robot.grad is not None
