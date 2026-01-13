"""Tests for dataset and batch shapes."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from asvspoof5_domain_invariant_cm.data.audio import (
    AudioCollator,
    collate_audio_batch,
    compute_attention_mask,
    crop_or_pad,
)


class TestCropOrPad:
    """Test waveform cropping and padding."""

    def test_pad_short_waveform(self):
        waveform = torch.randn(1, 1000)
        result = crop_or_pad(waveform, 2000, mode="eval")

        assert result.shape == (1, 2000)
        # Check padding is zeros
        assert torch.allclose(result[:, 1000:], torch.zeros(1, 1000))

    def test_crop_long_waveform_eval(self):
        waveform = torch.randn(1, 3000)
        result = crop_or_pad(waveform, 1000, mode="eval")

        assert result.shape == (1, 1000)
        # Center crop: should start at (3000-1000)//2 = 1000
        assert torch.allclose(result, waveform[:, 1000:2000])

    def test_crop_long_waveform_train(self):
        torch.manual_seed(42)
        waveform = torch.randn(1, 3000)
        result = crop_or_pad(waveform, 1000, mode="train")

        assert result.shape == (1, 1000)
        # Random crop: just check shape

    def test_exact_length_unchanged(self):
        waveform = torch.randn(1, 1000)
        result = crop_or_pad(waveform, 1000, mode="eval")

        assert result.shape == (1, 1000)
        assert torch.allclose(result, waveform)

    def test_1d_input(self):
        waveform = torch.randn(1000)
        result = crop_or_pad(waveform, 500, mode="eval")

        assert result.shape == (500,)


class TestAttentionMask:
    """Test attention mask computation."""

    def test_basic_mask(self):
        lengths = torch.tensor([3, 5, 2])
        mask = compute_attention_mask(lengths, max_len=5)

        expected = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ])

        assert torch.equal(mask, expected)

    def test_all_full_length(self):
        lengths = torch.tensor([5, 5, 5])
        mask = compute_attention_mask(lengths, max_len=5)

        assert mask.all()

    def test_empty_sequence(self):
        lengths = torch.tensor([0, 3])
        mask = compute_attention_mask(lengths, max_len=3)

        assert mask[0].sum() == 0
        assert mask[1].sum() == 3


class TestCollateAudioBatch:
    """Test audio batch collation."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch of varying lengths."""
        return [
            {
                "waveform": torch.randn(1, 1000),
                "y_task": 0,
                "y_codec": 1,
                "y_codec_q": 2,
                "flac_file": "test_001.flac",
            },
            {
                "waveform": torch.randn(1, 500),
                "y_task": 1,
                "y_codec": 0,
                "y_codec_q": 1,
                "flac_file": "test_002.flac",
            },
            {
                "waveform": torch.randn(1, 800),
                "y_task": 0,
                "y_codec": 2,
                "y_codec_q": 0,
                "flac_file": "test_003.flac",
            },
        ]

    def test_batch_shapes_dynamic(self, sample_batch):
        """Test shapes with dynamic length (pad to max)."""
        batch = collate_audio_batch(sample_batch, fixed_length=None, mode="eval")

        assert batch["waveform"].shape == (3, 1000)  # Padded to max
        assert batch["attention_mask"].shape == (3, 1000)
        assert batch["lengths"].shape == (3,)
        assert batch["y_task"].shape == (3,)
        assert batch["y_codec"].shape == (3,)
        assert batch["y_codec_q"].shape == (3,)

    def test_batch_shapes_fixed(self, sample_batch):
        """Test shapes with fixed length."""
        batch = collate_audio_batch(sample_batch, fixed_length=600, mode="eval")

        assert batch["waveform"].shape == (3, 600)
        assert batch["attention_mask"].shape == (3, 600)
        assert batch["lengths"].shape == (3,)

    def test_dtypes(self, sample_batch):
        """Test tensor dtypes."""
        batch = collate_audio_batch(sample_batch, fixed_length=None, mode="eval")

        assert batch["waveform"].dtype == torch.float32
        assert batch["attention_mask"].dtype == torch.bool
        assert batch["lengths"].dtype == torch.long
        assert batch["y_task"].dtype == torch.long
        assert batch["y_codec"].dtype == torch.long
        assert batch["y_codec_q"].dtype == torch.long

    def test_labels_preserved(self, sample_batch):
        """Test that labels are correctly preserved."""
        batch = collate_audio_batch(sample_batch, fixed_length=None, mode="eval")

        assert batch["y_task"].tolist() == [0, 1, 0]
        assert batch["y_codec"].tolist() == [1, 0, 2]
        assert batch["y_codec_q"].tolist() == [2, 1, 0]

    def test_metadata_preserved(self, sample_batch):
        """Test that metadata is preserved."""
        batch = collate_audio_batch(sample_batch, fixed_length=None, mode="eval")

        assert batch["metadata"]["flac_file"] == ["test_001.flac", "test_002.flac", "test_003.flac"]

    def test_attention_mask_correctness(self, sample_batch):
        """Test attention mask correctness."""
        batch = collate_audio_batch(sample_batch, fixed_length=None, mode="eval")

        # Check lengths
        assert batch["lengths"].tolist() == [1000, 500, 800]

        # Check mask sums match lengths
        assert batch["attention_mask"][0].sum() == 1000
        assert batch["attention_mask"][1].sum() == 500
        assert batch["attention_mask"][2].sum() == 800


class TestAudioCollator:
    """Test AudioCollator callable."""

    def test_collator_train_mode(self):
        collator = AudioCollator(fixed_length=500, mode="train")

        batch = [
            {"waveform": torch.randn(1, 1000), "y_task": 0, "y_codec": 0, "y_codec_q": 0},
            {"waveform": torch.randn(1, 800), "y_task": 1, "y_codec": 1, "y_codec_q": 1},
        ]

        result = collator(batch)

        assert result["waveform"].shape == (2, 500)

    def test_collator_eval_mode(self):
        collator = AudioCollator(fixed_length=500, mode="eval")

        batch = [
            {"waveform": torch.randn(1, 1000), "y_task": 0, "y_codec": 0, "y_codec_q": 0},
            {"waveform": torch.randn(1, 800), "y_task": 1, "y_codec": 1, "y_codec_q": 1},
        ]

        result = collator(batch)

        assert result["waveform"].shape == (2, 500)
