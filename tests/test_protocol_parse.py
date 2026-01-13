"""Tests for protocol parsing and manifest creation."""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from asvspoof5_domain_invariant_cm.data.asvspoof5 import (
    KEY_TO_LABEL,
    PROTOCOL_COLUMNS,
    load_protocol,
    normalize_domain_value,
)
from asvspoof5_domain_invariant_cm.utils.paths import (
    build_audio_path,
    get_audio_dir_for_prefix,
)


class TestLabelMapping:
    """Test label convention: bonafide=0, spoof=1."""

    def test_bonafide_is_zero(self):
        assert KEY_TO_LABEL["bonafide"] == 0

    def test_spoof_is_one(self):
        assert KEY_TO_LABEL["spoof"] == 1


class TestDomainNormalization:
    """Test domain value normalization.

    ASVspoof5 protocol files use different conventions:
    - Train/dev: CODEC="-", CODEC_Q="-" for uncoded
    - Eval: CODEC="-", CODEC_Q="0" for uncoded

    Both "-" and "0" (for CODEC_Q only) should normalize to "NONE".
    """

    def test_dash_becomes_none(self):
        """Both CODEC and CODEC_Q: '-' -> 'NONE'."""
        assert normalize_domain_value("-") == "NONE"
        assert normalize_domain_value("-", is_codec_q=False) == "NONE"
        assert normalize_domain_value("-", is_codec_q=True) == "NONE"

    def test_zero_becomes_none_for_codec_q(self):
        """CODEC_Q: '0' -> 'NONE' (eval uncoded convention)."""
        # For CODEC_Q, "0" means uncoded
        assert normalize_domain_value("0", is_codec_q=True) == "NONE"

    def test_zero_unchanged_for_codec(self):
        """CODEC: '0' should NOT become 'NONE' (not valid for CODEC anyway)."""
        # For CODEC, "0" is not a special value (though it shouldn't appear)
        assert normalize_domain_value("0", is_codec_q=False) == "0"

    def test_normal_value_unchanged(self):
        """Regular codec names should be preserved."""
        assert normalize_domain_value("opus") == "opus"
        assert normalize_domain_value("mp3") == "mp3"
        assert normalize_domain_value("C01") == "C01"
        assert normalize_domain_value("C11") == "C11"

    def test_quality_values_unchanged(self):
        """Quality values 1-8 should be preserved."""
        for q in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            assert normalize_domain_value(q, is_codec_q=True) == q

    def test_none_becomes_none_string(self):
        assert normalize_domain_value(None) == "NONE"
        assert normalize_domain_value(None, is_codec_q=True) == "NONE"

    def test_nan_becomes_none(self):
        import numpy as np
        assert normalize_domain_value(float("nan")) == "NONE"
        assert normalize_domain_value(float("nan"), is_codec_q=True) == "NONE"

    def test_train_dev_normalization(self):
        """Train/dev: CODEC='-', CODEC_Q='-' both -> 'NONE'."""
        # Simulating train/dev protocol values
        codec = "-"
        codec_q = "-"
        assert normalize_domain_value(codec, is_codec_q=False) == "NONE"
        assert normalize_domain_value(codec_q, is_codec_q=True) == "NONE"

    def test_eval_uncoded_normalization(self):
        """Eval uncoded: CODEC='-', CODEC_Q='0' both -> 'NONE'."""
        # Simulating eval protocol values for uncoded samples
        codec = "-"
        codec_q = "0"
        assert normalize_domain_value(codec, is_codec_q=False) == "NONE"
        assert normalize_domain_value(codec_q, is_codec_q=True) == "NONE"

    def test_eval_coded_preserved(self):
        """Eval coded: CODEC='C05', CODEC_Q='3' both preserved."""
        codec = "C05"
        codec_q = "3"
        assert normalize_domain_value(codec, is_codec_q=False) == "C05"
        assert normalize_domain_value(codec_q, is_codec_q=True) == "3"


class TestAudioPathPrefix:
    """Test filename prefix to audio directory mapping."""

    def test_t_prefix(self):
        assert get_audio_dir_for_prefix("T_") == "flac_T"

    def test_d_prefix(self):
        assert get_audio_dir_for_prefix("D_") == "flac_D"

    def test_e_prefix(self):
        assert get_audio_dir_for_prefix("E_") == "flac_E_eval"

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown filename prefix"):
            get_audio_dir_for_prefix("X_")


class TestBuildAudioPath:
    """Test absolute audio path construction."""

    def test_train_file_path(self):
        root = Path("/data/asvspoof5")
        path = build_audio_path("T_0001.flac", root)
        assert path == root / "flac_T" / "T_0001.flac"

    def test_dev_file_path(self):
        root = Path("/data/asvspoof5")
        path = build_audio_path("D_0001.flac", root)
        assert path == root / "flac_D" / "D_0001.flac"

    def test_eval_file_path(self):
        root = Path("/data/asvspoof5")
        path = build_audio_path("E_0001.flac", root)
        assert path == root / "flac_E_eval" / "E_0001.flac"


class TestProtocolParsing:
    """Test whitespace-separated protocol parsing."""

    @pytest.fixture
    def sample_protocol_content(self):
        """Sample protocol content with whitespace separation."""
        return """SPK001 T_0001.flac M opus 16 seed001 A01 spoof spoof tmp
SPK002 T_0002.flac F - - seed002 - - bonafide tmp
SPK003 T_0003.flac M mp3 32 seed003 A02 spoof spoof tmp"""

    def test_whitespace_parsing(self, sample_protocol_content):
        """Test that protocol is parsed correctly with whitespace separation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(sample_protocol_content)
            f.flush()

            df = load_protocol(Path(f.name))

            assert len(df) == 3
            assert list(df.columns) == PROTOCOL_COLUMNS

    def test_column_values(self, sample_protocol_content):
        """Test that column values are extracted correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(sample_protocol_content)
            f.flush()

            df = load_protocol(Path(f.name))

            # Check first row
            assert df.iloc[0]["speaker_id"] == "SPK001"
            assert df.iloc[0]["flac_file"] == "T_0001.flac"
            assert df.iloc[0]["gender"] == "M"
            assert df.iloc[0]["codec"] == "opus"
            assert df.iloc[0]["codec_q"] == "16"
            assert df.iloc[0]["key"] == "spoof"

            # Check bonafide row
            assert df.iloc[1]["key"] == "bonafide"
            assert df.iloc[1]["codec"] == "-"

    def test_dash_in_codec_fields(self, sample_protocol_content):
        """Test that dash values are preserved in parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(sample_protocol_content)
            f.flush()

            df = load_protocol(Path(f.name))

            # Row with dashes
            assert df.iloc[1]["codec"] == "-"
            assert df.iloc[1]["codec_q"] == "-"
