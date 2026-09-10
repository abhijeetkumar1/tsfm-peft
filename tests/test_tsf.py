"""Monash .tsf parsing, on small files written in the test."""

import numpy as np
import pytest

from tsfm_peft.data.tsf import normalise_timestamp, read_tsf

MINIMAL = """# a comment
@relation Demo
@attribute series_name string
@attribute start_timestamp date
@frequency daily
@horizon 56
@missing false
@equallength true
@data
T1:1996-03-18 00-00-00:1.0,2.0,3.0,
T2:1996-03-19 12-30-45:4,5,6
"""


def write(tmp_path, text, name="demo.tsf"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestReadTsf:
    def test_parses_metadata(self, tmp_path):
        parsed = read_tsf(write(tmp_path, MINIMAL))
        assert parsed.metadata["relation"] == "Demo"
        assert parsed.metadata["frequency"] == "daily"
        assert parsed.metadata["horizon"] == "56"

    def test_parses_attributes_in_order(self, tmp_path):
        parsed = read_tsf(write(tmp_path, MINIMAL))
        assert parsed.attribute_names == ("series_name", "start_timestamp")

    def test_parses_records(self, tmp_path):
        parsed = read_tsf(write(tmp_path, MINIMAL))
        assert len(parsed.records) == 2
        assert parsed.records[0].attributes["series_name"] == "T1"
        assert parsed.records[0].values.tolist() == [1.0, 2.0, 3.0]
        assert parsed.records[1].values.dtype == np.float64

    def test_ignores_a_trailing_comma(self, tmp_path):
        # Monash files end every value list with a comma; an empty final field is not a
        # missing observation.
        parsed = read_tsf(write(tmp_path, MINIMAL))
        assert parsed.records[0].values.size == 3

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        text = MINIMAL.replace("@data\n", "@data\n\n# mid-file comment\n")
        assert len(read_tsf(write(tmp_path, text)).records) == 2

    def test_missing_values_are_rejected_by_default(self, tmp_path):
        text = MINIMAL.replace("1.0,2.0,3.0,", "1.0,?,3.0,")
        with pytest.raises(ValueError, match="missing values"):
            read_tsf(write(tmp_path, text))

    def test_missing_values_become_nan_when_allowed(self, tmp_path):
        text = MINIMAL.replace("1.0,2.0,3.0,", "1.0,?,3.0,")
        parsed = read_tsf(write(tmp_path, text), allow_missing=True)
        assert np.isnan(parsed.records[0].values[1])

    def test_wrong_field_count_is_rejected(self, tmp_path):
        text = MINIMAL.replace("T2:1996-03-19 12-30-45:4,5,6", "T2:4,5,6")
        with pytest.raises(ValueError, match="colon-separated"):
            read_tsf(write(tmp_path, text))

    def test_empty_value_list_is_rejected(self, tmp_path):
        text = MINIMAL.replace("1.0,2.0,3.0,", "")
        with pytest.raises(ValueError, match="no values"):
            read_tsf(write(tmp_path, text))

    def test_data_before_the_data_section_is_rejected(self, tmp_path):
        text = "@relation Demo\nT1:1,2,3\n@data\n"
        with pytest.raises(ValueError, match="before @data"):
            read_tsf(write(tmp_path, text))

    def test_directive_after_the_data_section_is_rejected(self, tmp_path):
        text = MINIMAL + "@frequency hourly\n"
        with pytest.raises(ValueError, match="after @data"):
            read_tsf(write(tmp_path, text))

    def test_file_without_a_data_section_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no @data"):
            read_tsf(write(tmp_path, "@relation Demo\n@attribute series_name string\n"))

    def test_empty_data_section_is_rejected(self, tmp_path):
        text = MINIMAL.split("@data")[0] + "@data\n"
        with pytest.raises(ValueError, match="empty"):
            read_tsf(write(tmp_path, text))


class TestNormaliseTimestamp:
    def test_converts_the_monash_time_separator(self):
        assert normalise_timestamp("1996-03-18 00-00-00") == "1996-03-18T00:00:00"

    def test_passes_through_an_iso_timestamp(self):
        assert normalise_timestamp("2016-07-01 00:00:00") == "2016-07-01T00:00:00"

    def test_handles_a_date_without_a_time(self):
        assert normalise_timestamp("2016-07-01") == "2016-07-01"
