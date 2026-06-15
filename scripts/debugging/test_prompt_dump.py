"""Unit tests for the S2 prompt/image dumper (no torch / model needed).

Uses a fake processor that mirrors the real call pattern:
  apply_chat_template(conversation) -> text   (normal method)
  processor(text=[...], images=[...])          (dunder __call__ on the class)
  processor.tokenizer.decode(...) -> output    (normal method)
"""

import json
import os
import types

import pytest

from internnav.model.utils.prompt_dump import (
    PromptDumper,
    TrainingPromptDumper,
    prompt_dump_enabled,
)


class _FakeImage:
    def __init__(self, tag):
        self.tag = tag
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        with open(path, 'w') as f:
            f.write(self.tag)


class _FakeTokenizer:
    def decode(self, *args, **kwargs):
        return 'STOP'


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, conversation, **kwargs):
        # mimic real output: text with one <image> per image turn segment
        n_imgs = sum(
            1
            for turn in conversation
            for c in turn.get('content', [])
            if isinstance(c, dict) and c.get('type') == 'image'
        )
        return 'PROMPT ' + ('<image>' * n_imgs)

    def __call__(self, *args, **kwargs):
        return {'input_ids': [[1, 2, 3]]}


def _conversation(instruction, n_images):
    content = [{'type': 'text', 'text': instruction}]
    content += [{'type': 'image', 'image': _FakeImage(f'i{i}')} for i in range(n_images)]
    return [{'role': 'user', 'content': content}]


def test_enabled_flag():
    assert prompt_dump_enabled('prompt', '/tmp/x')
    assert prompt_dump_enabled('bev,prompt', '/tmp/x')
    assert not prompt_dump_enabled('bev', '/tmp/x')
    assert not prompt_dump_enabled('prompt', None)


def test_explicit_dump_writes_all_artifacts(tmp_path):
    proc = _FakeProcessor()
    dumper = PromptDumper(str(tmp_path))
    dumper.install(proc)

    conv = _conversation('go to the kitchen', n_images=2)
    text = proc.apply_chat_template(conv)
    images = [_FakeImage('a'), _FakeImage('b')]
    proc(text=[text], images=images)

    class _Out:
        output_pixel = None
        output_action = [1, 2, 3]

    dumper.dump(policy=None, output=_Out(), look_down=False)

    step_dir = tmp_path / 's2_step_000000'
    assert (step_dir / 'prompt.txt').read_text() == 'PROMPT <image><image>'
    assert (step_dir / 'sources.txt').read_text() == 'go to the kitchen'
    assert sorted(os.listdir(step_dir / 'images')) == ['00.jpg', '01.jpg']
    meta = json.loads((step_dir / 'meta.json').read_text())
    assert meta['num_input_images'] == 2
    assert meta['num_image_tokens'] == 2
    assert meta['output_action'] == [1, 2, 3]


def test_autodump_fires_on_decode(tmp_path):
    proc = _FakeProcessor()
    dumper = PromptDumper(str(tmp_path))

    class _Policy:
        episode_idx = 5
        llm_output = ''

    dumper.install_autodump(proc, _Policy())

    conv = _conversation('find the door', n_images=1)
    text = proc.apply_chat_template(conv)
    proc(text=[text], images=[_FakeImage('x')])
    out = proc.tokenizer.decode([0, 1, 2])  # should trigger one dump

    assert out == 'STOP'
    step_dir = tmp_path / 's2_step_000000'
    assert (step_dir / 'output.txt').read_text() == 'STOP'
    assert (step_dir / 'sources.txt').read_text() == 'find the door'
    meta = json.loads((step_dir / 'meta.json').read_text())
    assert meta['episode_idx'] == 5
    assert meta['num_input_images'] == 1

    # a second bare decode with no fresh prompt must NOT create another folder
    proc.tokenizer.decode([3])
    assert not (tmp_path / 's2_step_000001').exists()


def test_disabled_when_no_prompt_token(tmp_path):
    # The policy/evaluator only build a dumper when enabled; emulate that guard.
    assert prompt_dump_enabled('bev', str(tmp_path)) is False


# --------------------------------------------------------------- training side


class _FakeTensor:
    """Minimal stand-in for a [T,H,W,3] float image tensor."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_training_source_capture_and_dump(tmp_path):
    import numpy as np

    # fake dataset module exposing preprocess_qwen_2_visual
    calls = {'n': 0}

    def preprocess_qwen_2_visual(sources, tokenizer, **kwargs):
        calls['n'] += 1
        return {'input_ids': [[1, 2, 3]]}

    mod = types.SimpleNamespace(preprocess_qwen_2_visual=preprocess_qwen_2_visual)

    dumper = TrainingPromptDumper(str(tmp_path), max_samples=2)
    dumper.install_source_capture(mod)
    # idempotent + actually wrapped
    dumper.install_source_capture(mod)
    assert getattr(mod.preprocess_qwen_2_visual, '_prompt_dump_wrapped', False)

    chat_sources = [[
        {'from': 'human', 'value': 'Your task is to reach the sofa. <image>.'},
        {'from': 'gpt', 'value': 'STOP'},
    ]]
    mod.preprocess_qwen_2_visual(chat_sources, tokenizer=None)  # sets _last_sources
    assert calls['n'] == 1

    traj = _FakeTensor((np.ones((3, 8, 8, 3), dtype=np.float32)))
    dumper.maybe_dump(0, {'traj_images': traj})

    d = tmp_path / 'sample_000000'
    assert (d / 'sources.txt').read_text() == 'Your task is to reach the sofa. <image>.'
    assert '[human]' in (d / 'prompt.txt').read_text()
    assert '[gpt]' in (d / 'prompt.txt').read_text()
    assert sorted(os.listdir(d / 'images')) == ['00.jpg', '01.jpg', '02.jpg']
    meta = json.loads((d / 'meta.json').read_text())
    assert meta['num_traj_images'] == 3
    assert meta['num_image_tokens'] == 1


def test_training_respects_max_samples(tmp_path):
    dumper = TrainingPromptDumper(str(tmp_path), max_samples=2)
    dumper._last_sources = [[{'from': 'human', 'value': 'x <image>'}]]
    dumper.maybe_dump(5, {'traj_images': None})  # index >= max_samples → skip
    assert not (tmp_path / 'sample_000005').exists()
