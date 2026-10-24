import argparse
import logging
from pathlib import Path

import gentropy.common.stats
import pyarrow.parquet as pq
import pyspark.sql.functions as f
import pyspark.sql.types as t
from gentropy.common.genomic_region import GenomicRegion, KnownGenomicRegions
from gentropy.common.session import Session
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics

logger = logging.getLogger(__name__)

session = Session()
spark = session.spark

variant_ids = (
    spark.read.parquet("variant-ids")
    .filter(f.col("variantId").isNotNull())
    .select(
        "marker_name",
        "genome_build",
        "chromosome",
        "position",
        "variantId",
        "alternateAllele",
    )
    .cache()
)


def genome_build(parquet_path: Path) -> str:
    # Files without the key are the meta-analyses, which use the GRCh37 legends
    metadata = pq.read_metadata(parquet_path).metadata or {}
    return metadata.get(b"genome_build", b"GRCh37").decode()


def load(parquet_paths: list[Path]) -> SummaryStatistics:
    genome_builds = spark.createDataFrame(
        [(path.stem, genome_build(path)) for path in parquet_paths],
        ["studyId", "genome_build"],
    )
    is_alternate = f.upper(f.col("allele1")) == f.col("alternateAllele")
    data_frame = (
        spark.read.parquet(*(parquet_path.as_posix() for parquet_path in parquet_paths))
        # studyId is the file name without extension
        .withColumn(
            "studyId", f.regexp_extract(f.input_file_name(), r"([^/]+)\.parquet$", 1)
        )
        .join(f.broadcast(genome_builds), on="studyId")
        .join(f.broadcast(variant_ids), on=["marker_name", "genome_build"])
        .select(
            "studyId",
            "chromosome",
            "position",
            "variantId",
            f.when(is_alternate, f.col("freq1"))
            .otherwise(1 - f.col("freq1"))
            .cast(t.FloatType())
            .alias("effectAlleleFrequencyFromSource"),
            f.when(is_alternate, f.col("effect"))
            .otherwise(-f.col("effect"))
            .alias("beta"),
            f.col("std_err").alias("standardError"),
            f.col("sample_count").cast(t.IntegerType()).alias("sampleSize"),
            *gentropy.common.stats.split_pvalue_column(f.col("p_value")),
        )
    )
    return SummaryStatistics(_df=data_frame, _schema=SummaryStatistics.get_schema())


def clump(summary_statistics: SummaryStatistics) -> StudyLocus:
    study_locus = summary_statistics.locus_breaker_clumping().exclude_region(
        GenomicRegion.from_known_genomic_region(KnownGenomicRegions.MHC)
    )
    return study_locus.annotate_locus_statistics_boundaries(summary_statistics)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Collect summary statistics")
    argument_parser.add_argument(
        "--phenotype-file", type=Path, nargs="+", required=True
    )
    argument_parser.add_argument("--input-path", type=Path, required=True)

    arguments = argument_parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    phenotypes_set: set[str] = {
        row.phenotype
        for row in session.spark.read.parquet(
            *(phenotype_file.as_posix() for phenotype_file in arguments.phenotype_file)
        )
        .select("phenotype")
        .dropDuplicates()
        .collect()
    }
    logger.info(f"Found {len(phenotypes_set)} phenotypes")

    parquet_paths = [
        path
        for path in arguments.input_path.rglob("*.parquet")
        if path.stem in phenotypes_set
    ]
    logger.info(f"Found {len(parquet_paths)} parquet files")

    summary_statistics = load(parquet_paths)
    study_locus = clump(summary_statistics)
    study_locus.df.coalesce(1).write.mode("append").parquet("study-locus")


if __name__ == "__main__":
    main()
