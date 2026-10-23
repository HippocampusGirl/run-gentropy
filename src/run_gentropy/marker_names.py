import argparse
from pathlib import Path

from gentropy.common.session import Session

session = Session()
spark = session.spark

marker_names_path = Path("marker-names")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Collect the union of marker names"
    )
    argument_parser.add_argument("input_path", type=Path)

    arguments = argument_parser.parse_args()

    parquet_paths = sorted(arguments.input_path.rglob("*.parquet"))

    data_frame = (
        spark.read.parquet(*(parquet_path.as_posix() for parquet_path in parquet_paths))
        .select("marker_name")
        .dropna()
    )
    
    if marker_names_path.exists():
        data_frame = data_frame.unionByName(
            spark.read.parquet(marker_names_path.as_posix()).select("marker_name")
        )

    (
        data_frame.distinct()
        .repartition(1)
        .localCheckpoint(eager=True)
        .write.mode("overwrite")
        .parquet(marker_names_path.as_posix())
    )


if __name__ == "__main__":
    main()
