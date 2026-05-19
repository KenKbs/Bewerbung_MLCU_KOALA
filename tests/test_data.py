from wahlumfragen.data import CSV_COLUMNS, PARTY_COLUMNS, generate_sample_polls, load_poll_csv, write_sample_csv


def test_generate_sample_polls_are_valid():
    """Test that synthetic polls have the expected shape."""
    polls = generate_sample_polls()
    assert len(polls) == 12

    for poll in polls:
        party_total = sum(getattr(poll, party) for party in PARTY_COLUMNS)
        assert party_total == 100.0


def test_write_and_load_sample_csv(tmp_path):
    """Test writing and loading the sample poll CSV."""
    output_path = write_sample_csv(tmp_path / "sample.csv")
    rows = load_poll_csv(output_path)

    assert output_path.exists()
    assert len(rows) == 12
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert rows[0]["sample_size"] == 1006
    assert rows[0]["cdu_csu"] == 28.0
