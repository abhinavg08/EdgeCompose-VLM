import pytest
import torch

from edgecompose.compression import UniformCompression, VisionFeatures, build_compressor
from edgecompose.compression.visionzip import VisionZipCompression, attention_importance_and_keys


def _feats(n=100, d=16, dk=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return VisionFeatures(
        embeds=torch.randn(n, d, generator=g),
        grid_thw=torch.tensor([[1, 20, 20]]),
        importance=torch.rand(n, generator=g),
        keys=torch.randn(n, dk, generator=g),
    )


@pytest.mark.parametrize("r,expected", [(0.75, 75), (0.5, 50), (0.25, 25)])
def test_visionzip_budget_matches_reference_split(r, expected):
    f = _feats()
    res = VisionZipCompression(r).compress(f)
    assert res.keep_indices.numel() == expected
    assert res.num_contextual == 5  # max(int(0.05 * 100), 1)
    assert res.num_dominant == expected - 5
    assert torch.all(res.keep_indices[1:] > res.keep_indices[:-1])  # sorted raster order, unique


def test_visionzip_keeps_top_attention_tokens_unchanged():
    f = _feats()
    res = VisionZipCompression(0.5).compress(f)
    top = torch.topk(f.importance, res.num_dominant).indices
    kept = set(res.keep_indices.tolist())
    assert set(top.tolist()) <= kept
    pos = {int(i): j for j, i in enumerate(res.keep_indices.tolist())}
    for i in top.tolist():
        assert torch.equal(res.embeds[pos[i]], f.embeds[i])  # dominant tokens are not modified


def test_visionzip_contextual_tokens_are_merged():
    f = _feats()
    res = VisionZipCompression(0.5).compress(f)
    top = set(torch.topk(f.importance, res.num_dominant).indices.tolist())
    ctx = [i for i in res.keep_indices.tolist() if i not in top]
    assert len(ctx) == res.num_contextual
    pos = {int(i): j for j, i in enumerate(res.keep_indices.tolist())}
    changed = [not torch.allclose(res.embeds[pos[i]], f.embeds[i]) for i in ctx]
    assert any(changed)


def test_identity_at_full_retention():
    f = _feats()
    res = build_compressor("visionzip", 1.0).compress(f)
    assert res.keep_indices.numel() == 100
    assert torch.equal(res.embeds, f.embeds)


def test_uniform_is_deterministic_and_spread():
    f = _feats()
    a = UniformCompression(0.25).compress(f)
    b = UniformCompression(0.25).compress(f)
    assert torch.equal(a.keep_indices, b.keep_indices)
    assert a.keep_indices[0] == 0 and a.keep_indices[-1] == 99
    assert a.keep_indices.numel() == 25


def test_attention_stats_reorder_to_raster():
    # 2 merged tokens x 4 patches, 1 head. Window order = [1, 0] (token 1 first).
    s, h, d = 8, 1, 4
    q = torch.zeros(s, h, d)
    k = torch.zeros(s, h, d)
    k[:4, 0, 0] = 5.0  # patches of window-position 0 (= raster token 1) receive attention
    q[:, 0, 0] = 1.0
    window_index = torch.tensor([1, 0])
    imp, keys = attention_importance_and_keys(q, k, window_index, merge_unit=4, scale=1.0)
    assert imp.shape == (2,) and keys.shape == (2, d)
    assert imp[1] > imp[0]  # raster token 1 is the attended one
    assert keys[1, 0] == pytest.approx(5.0) and keys[0, 0] == pytest.approx(0.0)


def test_attention_stats_segments_isolate_images():
    # two images of 1 merged token (4 patches) each; with segments, image 0's statistics must
    # not depend on image 1's keys.
    torch.manual_seed(0)
    q = torch.randn(8, 2, 4)
    k = torch.randn(8, 2, 4)
    wi = torch.tensor([0, 1])
    imp_a, _ = attention_importance_and_keys(q, k, wi, segments=[(0, 4), (4, 8)])
    k2 = k.clone()
    k2[4:] = torch.randn(4, 2, 4) * 10
    imp_b, _ = attention_importance_and_keys(q, k2, wi, segments=[(0, 4), (4, 8)])
    assert imp_a[0] == pytest.approx(imp_b[0])
    # each segment's column sums add up to its number of queries (softmax rows sum to 1)
    assert float(imp_a.sum()) == pytest.approx(2 * 4 / 4)  # mean over 4 patches per token


def test_invalid_retention():
    with pytest.raises(ValueError):
        VisionZipCompression(0.0)
    with pytest.raises(ValueError):
        build_compressor("magic", 0.5)
