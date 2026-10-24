import argparse
import logging
from pathlib import Path

import gentropy.common.stats
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pyspark.sql.functions as f
import pyspark.sql.types as t
from gentropy.common.genomic_region import GenomicRegion, KnownGenomicRegions
from gentropy.common.session import Session
from gentropy.config import LocusBreakerClumpingConfig
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics
from pyspark.sql import DataFrame

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


def load(parquet_paths: list[Path]) -> DataFrame:
    genome_builds = spark.createDataFrame(
        [(path.stem, genome_build(path)) for path in parquet_paths],
        ["studyId", "genome_build"],
    )
    return (
        spark.read.parquet(*(parquet_path.as_posix() for parquet_path in parquet_paths))
        # studyId is the file name without extension
        .withColumn(
            "studyId", f.regexp_extract(f.input_file_name(), r"([^/]+)\.parquet$", 1)
        )
        .join(f.broadcast(genome_builds), on="studyId")
    )


def to_summary_statistics(data_frame: DataFrame) -> SummaryStatistics:
    is_alternate = f.upper(f.col("allele1")) == f.col("alternateAllele")
    data_frame = data_frame.select(
        "studyId",
        "chromosome",
        "position",
        "variantId",
        f.when(is_alternate, f.col("freq1"))
        .otherwise(1 - f.col("freq1"))
        .cast(t.FloatType())
        .alias("effectAlleleFrequencyFromSource"),
        f.when(is_alternate, f.col("effect")).otherwise(-f.col("effect")).alias("beta"),
        f.col("std_err").alias("standardError"),
        f.col("sample_count").cast(t.IntegerType()).alias("sampleSize"),
        *gentropy.common.stats.split_pvalue_column(f.col("p_value")),
    )
    return SummaryStatistics(_df=data_frame, _schema=SummaryStatistics.get_schema())


def clump(data_frame: DataFrame) -> StudyLocus:
    significant = data_frame.filter(
        f.col("p_value").cast(t.DoubleType())
        <= LocusBreakerClumpingConfig.lbc_baseline_pvalue
    )
    study_locus = (
        to_summary_statistics(
            variant_ids.join(f.broadcast(significant), on=["marker_name", "genome_build"])
        )
        .locus_breaker_clumping()
        .exclude_region(
            GenomicRegion.from_known_genomic_region(KnownGenomicRegions.MHC)
        )
        .persist()
    )

    loci = study_locus.df.select(
        f.col("chromosome").alias("locusChromosome"), "locusStart", "locusEnd"
    ).distinct()
    locus_variants = variant_ids.join(
        f.broadcast(loci),
        on=(f.col("chromosome") == f.col("locusChromosome"))
        & f.col("position").between(f.col("locusStart"), f.col("locusEnd")),
        how="left_semi",
    )
    return study_locus.annotate_locus_statistics_boundaries(
        to_summary_statistics(
            data_frame.join(f.broadcast(locus_variants), on=["marker_name", "genome_build"])
        )
    )


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

    study_locus = clump(load(parquet_paths))
    study_locus.df.coalesce(1).write.mode("append").parquet("study-locus")


if __name__ == "__main__":
    main()
