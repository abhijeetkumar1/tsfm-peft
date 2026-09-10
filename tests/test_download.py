"""Caching, checksum pinning and atomic writes. No network: sources are file:// URLs."""

import hashlib
import zipfile

import pytest

from tsfm_peft.data import download as dl
from tsfm_peft.data.download import (
    ChecksumMismatchError,
    cached_download,
    extract_member,
    sha256_of,
)

PAYLOAD = b"a,b\n1,2\n3,4\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(PAYLOAD)
    return path


class TestSha256:
    def test_matches_hashlib(self, source):
        assert sha256_of(source) == DIGEST

    def test_reads_files_larger_than_one_chunk(self, tmp_path):
        big = tmp_path / "big.bin"
        blob = b"x" * (3 * (1 << 20) + 17)
        big.write_bytes(blob)
        assert sha256_of(big) == hashlib.sha256(blob).hexdigest()


class TestCachedDownload:
    def test_fetches_and_verifies(self, source, tmp_path):
        dest = tmp_path / "cache" / "file.csv"
        assert cached_download(source.as_uri(), dest, sha256=DIGEST) == dest
        assert dest.read_bytes() == PAYLOAD

    def test_second_call_uses_the_cache(self, source, tmp_path, monkeypatch):
        dest = tmp_path / "file.csv"
        cached_download(source.as_uri(), dest, sha256=DIGEST)

        def explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("cache hit should not re-fetch")

        monkeypatch.setattr(dl, "_fetch", explode)
        assert cached_download(source.as_uri(), dest, sha256=DIGEST) == dest

    def test_force_re_fetches(self, source, tmp_path):
        dest = tmp_path / "file.csv"
        cached_download(source.as_uri(), dest, sha256=DIGEST)
        source.write_bytes(PAYLOAD)  # same bytes, so the checksum still holds
        assert cached_download(source.as_uri(), dest, sha256=DIGEST, force=True) == dest

    def test_corrupt_cache_entry_is_replaced(self, source, tmp_path):
        dest = tmp_path / "file.csv"
        dest.write_bytes(b"truncated")
        assert cached_download(source.as_uri(), dest, sha256=DIGEST) == dest
        assert dest.read_bytes() == PAYLOAD

    def test_upstream_change_raises_and_leaves_no_file(self, source, tmp_path):
        dest = tmp_path / "file.csv"
        wrong = "0" * 64
        with pytest.raises(ChecksumMismatchError, match="pinned checksum"):
            cached_download(source.as_uri(), dest, sha256=wrong)
        assert not dest.exists()

    def test_mismatch_message_names_both_digests(self, source, tmp_path):
        with pytest.raises(ChecksumMismatchError, match=DIGEST):
            cached_download(source.as_uri(), tmp_path / "f", sha256="0" * 64)

    def test_no_partial_files_are_left_behind(self, source, tmp_path):
        dest = tmp_path / "cache" / "file.csv"
        cached_download(source.as_uri(), dest, sha256=DIGEST)
        assert [p.name for p in dest.parent.iterdir()] == ["file.csv"]

    def test_a_failed_fetch_leaves_no_partial_file(self, tmp_path):
        dest = tmp_path / "cache" / "file.csv"
        missing = (tmp_path / "does-not-exist").as_uri()
        with pytest.raises(OSError, match=r"."):
            cached_download(missing, dest, sha256=DIGEST)
        assert not dest.exists()
        assert list(dest.parent.iterdir()) == []


class TestExtractMember:
    @pytest.fixture
    def archive(self, tmp_path):
        path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("inner.tsf", "hello")
            zf.writestr("other.txt", "ignored")
        return path

    def test_extracts_only_the_named_member(self, archive, tmp_path):
        dest = extract_member(archive, "inner.tsf", tmp_path / "out" / "inner.tsf")
        assert dest.read_text() == "hello"
        assert [p.name for p in dest.parent.iterdir()] == ["inner.tsf"]

    def test_is_idempotent(self, archive, tmp_path):
        dest = tmp_path / "inner.tsf"
        extract_member(archive, "inner.tsf", dest)
        extract_member(archive, "inner.tsf", dest)
        assert dest.read_text() == "hello"

    def test_missing_member_raises(self, archive, tmp_path):
        with pytest.raises(KeyError):
            extract_member(archive, "absent.tsf", tmp_path / "absent.tsf")
