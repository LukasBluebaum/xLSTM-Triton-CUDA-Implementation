import torch


def check(*inputs, atol=1e-2, rtol=1e-2):
    for i, (a, b) in enumerate(zip(inputs[::2], inputs[1::2])):
        c = torch.allclose(a.cpu(), b.cpu(), atol=atol, rtol=rtol)
        c1 = torch.isclose(a.cpu(), b.cpu(), atol=atol, rtol=rtol)
        assert c, (
            f"{i}\n{a}\n{b}\n{c1}\nNon-Matching Values A:\n{a[c1 == False]}\n"
            f"Non-Matching Values B:\n{b[c1 == False]}\n Percentage non-matching: "
            f"{format(torch.sum(c1 == False) / torch.numel(a), '.10f')}"
        )
    print("EQUAL")
