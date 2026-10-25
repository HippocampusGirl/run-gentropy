import argparse
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pyspark.sql.functions as f
from gentropy.common.session import Session

session = Session()
spark = session.spark

marker_names_path = Path("marker-names")


def genome_build(parquet_path: Path) -> str:
    # Files without the key are the meta-analyses, which use the GRCh37 legends
    metadata = pq.read_metadata(parquet_path).metadata or {}
    return metadata.get(b"genome_build", b"GRCh37").decode()


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Collect the union of marker names"
    )
    argument_parser.add_argument("input_path", type=Path)

    arguments = argument_parser.parse_args()

    parquet_paths = sorted(arguments.input_path.rglob("*.parquet"))
    # The same marker name is a different variant in GRCh37 and GRCh38
    genome_builds = spark.createDataFrame(
        [(path.stem, genome_build(path)) for path in parquet_paths],
        ["stem", "genome_build"],
    )

    data_frame = (
        spark.read.parquet(*(parquet_path.as_posix() for parquet_path in parquet_paths))
        .withColumn(
            "stem", f.regexp_extract(f.input_file_name(), r"([^/]+)\.parquet$", 1)
        )
        .join(f.broadcast(genome_builds), on="stem")
        .select("marker_name", "genome_build")
        .dropna()
    )

    if marker_names_path.exists():
        existing = spark.read.parquet(marker_names_path.as_posix())

        # Add column if not available
        if "genome_build" not in existing.columns:
            existing = existing.withColumn("genome_build", f.lit("GRCh37"))

        data_frame = data_frame.unionByName(
            existing.select("marker_name", "genome_build")
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
