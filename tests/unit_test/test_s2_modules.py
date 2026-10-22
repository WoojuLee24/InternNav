"""
Unit tests for S2 modules: latent_action, latent_reasoning, world_predictor, s2_kv_cache.

Run with: pytest tests/unit_test/test_s2_modules.py -v
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent.parent


def _load(name: str):
    path = ROOT / "internnav" / "agent" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ─── latent_action ────────────────────────────────────────────────────────────

class TestActionTokenVocabulary:
    @pytest.fixture
    def vocab(self):
        m = _load("latent_action")
        return m.build_default_vocab(device="cpu")

    def test_token_to_action_shape(self, vocab):
        action = vocab.token_to_action(1)
        assert action.shape == (3,)

    def test_special_token_stop(self, vocab):
        action = vocab.token_to_action(0)  # STOP
        # Should decode to near-zero (but not exact due to untrained decoder)
        assert action.shape == (3,)

    def test_vq_loss_runs(self, vocab):
        actions = torch.randn(4, 3)
        loss, tids = vocab.vq_loss(actions)
        assert loss.item() >= 0
        assert len(tids) == 4

    def test_vq_loss_decreases(self, vocab):
        """Loss should decrease with a gradient step."""
        opt = torch.optim.Adam(vocab.parameters(), lr=1e-3)
        actions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.57]])
        loss0, _ = vocab.vq_loss(actions)
        for _ in range(10):
            opt.zero_grad()
            loss, _ = vocab.vq_loss(actions)
            loss.backward()
            opt.step()
        assert loss.item() < loss0.item() + 0.1  # should not explode

    def test_round_trip_token(self, vocab):
        """action_to_nearest_token on a codebook entry should return itself."""
        action = vocab.token_to_action(5)
        tid = vocab.action_to_nearest_token(action)
        assert isinstance(tid, int)

    def test_tokens_to_actions_batch(self, vocab):
        ids = torch.tensor([1, 2, 5])
        actions = vocab.tokens_to_actions(ids)
        assert actions.shape == (3, 3)


class TestLatentActionDecoder:
    @pytest.fixture
    def decoder(self):
        m = _load("latent_action")
        vocab = m.build_default_vocab()
        return m.LatentActionDecoder(vocab, max_tokens=3)

    def test_decode_plan_truncates_at_stop(self, decoder):
        wp = decoder.decode_plan([1, 2, 0, 5])  # STOP at index 2
        assert len(wp) == 2

    def test_decode_plan_empty_on_stop_first(self, decoder):
        wp = decoder.decode_plan([0, 1, 2])
        assert len(wp) == 0

    def test_cumulative_trajectory(self, decoder):
        traj = decoder.decode_cumulative_trajectory([1, 4])
        assert traj.shape == (2, 3)


# ─── latent_reasoning ─────────────────────────────────────────────────────────

class TestLatentReasoner:
    @pytest.fixture
    def reasoner(self):
        m = _load("latent_reasoning")
        return m.LatentReasoner(thought_dim=64, context_dim=64, n_steps=2, vocab_size=32, max_decode_len=3)

    def test_forward_shapes(self, reasoner):
        ctx = torch.randn(1, 64)
        logits, thought, conf = reasoner(ctx)
        assert logits.shape == (1, 3, 32)
        assert thought.shape == (1, 64)
        assert conf.shape == (1, 1)

    def test_prev_thought_carry(self, reasoner):
        ctx = torch.randn(1, 64)
        _, thought1, _ = reasoner(ctx)
        _, thought2, _ = reasoner(ctx, prev_thought=thought1)
        assert thought2.shape == thought1.shape

    def test_predict_tokens_list(self, reasoner):
        ctx = torch.randn(1, 64)
        tokens, thought, conf = reasoner.predict_tokens(ctx)
        assert isinstance(tokens, list)
        assert len(tokens) <= 3
        assert isinstance(conf, float)


class TestS2ThoughtCache:
    @pytest.fixture
    def cache(self):
        m = _load("latent_reasoning")
        return m.S2ThoughtCache(thought_dim=64)

    def test_initial_drift_is_one(self, cache):
        assert cache.thought_drift() == 1.0

    def test_drift_after_identical_thoughts(self, cache):
        t = torch.ones(1, 64)
        cache.update(t)
        cache.update(t)
        assert cache.thought_drift() < 0.01  # cosine_sim ≈ 1, drift ≈ 0

    def test_reset_clears(self, cache):
        cache.update(torch.randn(1, 64))
        cache.reset()
        assert cache.current_thought is None
        assert cache._step == 0


# ─── world_predictor ──────────────────────────────────────────────────────────

class TestWorldPredictor:
    @pytest.fixture
    def wp(self):
        m = _load("world_predictor")
        return m.WorldPredictor(feature_dim=64, device="cpu")

    def test_predict_shape(self, wp):
        frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        pose = torch.tensor([1.0, 0.0, 0.0])
        score, feats = wp.predict_reachability(frame, pose)
        assert 0.0 <= score <= 1.0
        assert feats.shape == (64,)

    def test_online_update(self, wp):
        for _ in range(10):
            frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
            pose = torch.tensor([1.0, 0.0, 0.0])
            wp.update(frame, pose, reward=float(_ % 2))
        assert wp._stats["updates"] >= 1

    def test_stats_str(self, wp):
        s = wp.stats_str()
        assert "WorldPredictor" in s


# ─── s2_kv_cache ──────────────────────────────────────────────────────────────

class TestS2KVCache:
    @pytest.fixture
    def cache(self):
        m = _load("s2_kv_cache")
        return m.S2KVCache(max_history=4, global_memory_size=8)

    def test_first_call_is_miss(self, cache):
        prompt, reuse = cache.build_prompt("go to kitchen", "I see a hallway", None)
        assert reuse == 0
        assert "go to kitchen" in prompt

    def test_same_call_reuses_with_past_kv(self, cache):
        cache.update("hallway frame")
        prompt, reuse1 = cache.build_prompt("go to kitchen", "frame A", object())
        cache._last_prefix_hash = cache._prefix_hash("go to kitchen")
        _, reuse2 = cache.build_prompt("go to kitchen", "frame B", object())
        assert reuse2 > 0

    def test_global_memory_eviction(self, cache):
        for i in range(12):
            cache.update(f"frame {i}", place_id=f"place_{i}")
        assert len(cache._global_mem) <= 8

    def test_reset_clears(self, cache):
        cache.update("frame A", place_id="p1")
        cache.reset()
        assert len(cache._history) == 0
        assert len(cache._global_mem) == 0
