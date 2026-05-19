import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer

# Define constants
SAMPLE_FILENAME = "sample.csv"
DEFAULT_SAMPLE_PATH = Path("data") / SAMPLE_FILENAME
METADATA_COLUMNS = (
    "poll_id",
    "election",
    "region",
    "pollster",
    "sponsor",
    "fieldwork_start",
    "fieldwork_end",
    "published_at",
    "sample_size",
    "mode",
)
PARTY_COLUMNS = ("cdu_csu", "spd", "gruene", "fdp", "linke", "afd", "bsw", "sonstige")
CSV_COLUMNS = METADATA_COLUMNS + PARTY_COLUMNS


@dataclass(frozen=True)
class PollRecord:
    """Single synthetic poll observation in wide CSV format.
       Nothing fancy, just manual poll insertation. 
       Would be replaced by scraping in real project"""

    poll_id: str
    election: str
    region: str
    pollster: str
    sponsor: str
    fieldwork_start: str
    fieldwork_end: str
    published_at: str
    sample_size: int
    mode: str
    cdu_csu: float
    spd: float
    gruene: float
    fdp: float
    linke: float
    afd: float
    bsw: float
    sonstige: float

    def as_csv_row(self) -> dict[str, Any]:
        """Return the poll as a row compatible with csv.DictWriter."""
        return asdict(self)


def generate_sample_polls() -> list[PollRecord]:
    """Create synthetic Bundestag poll data for the prototype.

    The numbers are not real polling data. They are designed to make the later
    prototype interesting: FDP and BSW hover around the 5 percent threshold,
    while the larger parties have plausible positive and negative trade-offs.
    """
    return [
        PollRecord(
            "poll-2026-03-07-allensbach",
            "Bundestag",
            "Deutschland",
            "Allensbach",
            "synthetic",
            "2026-03-01",
            "2026-03-06",
            "2026-03-07",
            1006,
            "mixed",
            28.0,
            15.0,
            12.0,
            4.5,
            5.0,
            23.0,
            4.5,
            8.0,
        ),
        PollRecord(
            "poll-2026-03-15-insa",
            "Bundestag",
            "Deutschland",
            "INSA",
            "synthetic",
            "2026-03-10",
            "2026-03-14",
            "2026-03-15",
            1204,
            "online",
            27.0,
            16.0,
            11.5,
            4.0,
            5.5,
            23.5,
            5.0,
            7.5,
        ),
        PollRecord(
            "poll-2026-03-22-forsa",
            "Bundestag",
            "Deutschland",
            "Forsa",
            "synthetic",
            "2026-03-16",
            "2026-03-21",
            "2026-03-22",
            1502,
            "phone",
            26.5,
            15.5,
            12.5,
            4.0,
            6.0,
            24.0,
            4.5,
            7.0,
        ),
        PollRecord(
            "poll-2026-03-30-fg-wahlen",
            "Bundestag",
            "Deutschland",
            "Forschungsgruppe Wahlen",
            "synthetic",
            "2026-03-24",
            "2026-03-29",
            "2026-03-30",
            1376,
            "phone",
            28.5,
            15.5,
            11.0,
            4.5,
            5.0,
            22.5,
            5.0,
            8.0,
        ),
        PollRecord(
            "poll-2026-04-06-kantar",
            "Bundestag",
            "Deutschland",
            "Kantar",
            "synthetic",
            "2026-03-31",
            "2026-04-05",
            "2026-04-06",
            1411,
            "online",
            27.5,
            16.0,
            12.0,
            4.5,
            5.5,
            22.5,
            5.0,
            7.0,
        ),
        PollRecord(
            "poll-2026-04-12-yougov",
            "Bundestag",
            "Deutschland",
            "YouGov",
            "synthetic",
            "2026-04-07",
            "2026-04-11",
            "2026-04-12",
            2018,
            "online",
            26.0,
            15.5,
            12.5,
            4.0,
            6.0,
            24.5,
            5.0,
            6.5,
        ),
        PollRecord(
            "poll-2026-04-18-gms",
            "Bundestag",
            "Deutschland",
            "GMS",
            "synthetic",
            "2026-04-13",
            "2026-04-17",
            "2026-04-18",
            1003,
            "mixed",
            28.0,
            15.0,
            11.5,
            4.5,
            5.0,
            23.5,
            4.5,
            8.0,
        ),
        PollRecord(
            "poll-2026-04-25-infratest-dimap",
            "Bundestag",
            "Deutschland",
            "Infratest dimap",
            "synthetic",
            "2026-04-20",
            "2026-04-24",
            "2026-04-25",
            1321,
            "phone",
            27.0,
            16.0,
            12.5,
            4.0,
            5.5,
            23.0,
            5.0,
            7.0,
        ),
        PollRecord(
            "poll-2026-05-02-forsa",
            "Bundestag",
            "Deutschland",
            "Forsa",
            "synthetic",
            "2026-04-27",
            "2026-05-01",
            "2026-05-02",
            1504,
            "phone",
            26.5,
            15.0,
            12.0,
            4.0,
            6.0,
            24.0,
            5.5,
            7.0,
        ),
        PollRecord(
            "poll-2026-05-08-insa",
            "Bundestag",
            "Deutschland",
            "INSA",
            "synthetic",
            "2026-05-03",
            "2026-05-07",
            "2026-05-08",
            1201,
            "online",
            27.0,
            15.5,
            11.5,
            4.5,
            5.5,
            24.0,
            5.0,
            7.0,
        ),
        PollRecord(
            "poll-2026-05-13-fg-wahlen",
            "Bundestag",
            "Deutschland",
            "Forschungsgruppe Wahlen",
            "synthetic",
            "2026-05-08",
            "2026-05-12",
            "2026-05-13",
            1275,
            "phone",
            28.0,
            15.0,
            12.0,
            4.0,
            5.5,
            23.0,
            5.5,
            7.0,
        ),
        PollRecord(
            "poll-2026-05-17-allensbach",
            "Bundestag",
            "Deutschland",
            "Allensbach",
            "synthetic",
            "2026-05-12",
            "2026-05-16",
            "2026-05-17",
            1022,
            "mixed",
            27.5,
            15.5,
            12.0,
            4.5,
            5.0,
            23.5,
            5.0,
            7.0,
        ),
    ]


def validate_poll(record: PollRecord) -> None:
    """Validate one synthetic poll record before writing it to disk."""
    total = sum(float(getattr(record, party)) for party in PARTY_COLUMNS)
    if round(total, 6) != 100.0:
        msg = f"{record.poll_id} sums to {total}, expected 100.0"
        raise ValueError(msg)

    if record.sample_size <= 0:
        msg = f"{record.poll_id} has non-positive sample_size={record.sample_size}"
        raise ValueError(msg)


def write_sample_csv(output_path: Path | str = DEFAULT_SAMPLE_PATH) -> Path:
    """Write the synthetic sample poll CSV and return its path."""
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_sample_polls()
    for record in records:
        validate_poll(record)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(record.as_csv_row() for record in records)

    return csv_path


def load_poll_csv(data_path: Path | str) -> list[dict[str, Any]]:
    """Load poll rows from a CSV file or from all CSV files in a directory."""
    path = Path(data_path)
    if path.is_file():
        csv_files = [path]
    elif path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
    else:
        return []

    rows: list[dict[str, Any]] = []
    for csv_file in csv_files:
        with csv_file.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(_parse_poll_row(row))
    return rows


def _parse_poll_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse one CSV row into Python types."""
    parsed: dict[str, Any] = dict(row)
    parsed["sample_size"] = int(parsed["sample_size"])
    for party in PARTY_COLUMNS:
        parsed[party] = float(parsed[party])
    return parsed


def main(
    output_path: Path = typer.Argument(DEFAULT_SAMPLE_PATH, help="CSV file to generate."),
) -> None:
    """Generate the synthetic sample poll CSV."""
    csv_path = write_sample_csv(output_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    typer.run(main)
