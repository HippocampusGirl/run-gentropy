import argparse
import logging
from pathlib import Path

import gentropy.common.stats
import pyspark.sql.functions as f
import pyspark.sql.types as t
from gentropy.common.genomic_region import (
    GenomicRegion,
    KnownGenomicRegions,
    LiftOverSpark,
)
from gentropy.common.session import Session
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.summary_statistics import SummaryStatistics
from pyspark.sql.column import Column

logger = logging.getLogger(__name__)

session = Session()
spark = session.spark
liftover = LiftOverSpark(chain_file="data/grch37_to_grch38.over.chain.gz")


def allele_hash(reference: Column, alternate: Column) -> Column:
    """Order-invariant 64-bit hash of an allele pair: hash(A) + hash(B) == hash(B) + hash(A)."""
    return (
        f.xxhash64(reference.cast(t.StringType()))
        + f.xxhash64(alternate.cast(t.StringType()))
    ).alias("allele_hash")


variants = (
    spark.read.parquet("data/variant/*.parquet")
    .select(
        "chromosome",
        "position",
        allele_hash(f.col("referenceAllele"), f.col("alternateAllele")),
        "variantId",
        "alternateAllele",
    )
    .cache()
)


def load(parquet_paths: list[Path]) -> SummaryStatistics:
    split = f.split("marker_name", ":", limit=4)
    chromosome = split.getItem(0)
    position = split.getItem(1).cast(t.IntegerType())
    data_frame = (
        spark.read.parquet(*(parquet_path.as_posix() for parquet_path in parquet_paths))
        # studyId is the file name without extension
        .withColumn(
            "studyId", f.regexp_extract(f.input_file_name(), r"([^/]+)\.parquet$", 1)
        )
        .filter(f.col("p_value") <= 1e-5)
        .withColumn(
            "lifted_coordinates",
            f.explode(liftover.liftover_udf(chromosome, position)),
        )
        .select(
            "studyId",
            # Split marker_name
            f.regexp_replace(f.col("lifted_coordinates").getItem(0), "^chr", "").alias(
                "chromosome"
            ),
            f.col("lifted_coordinates")
            .getItem(1)
            .cast(t.IntegerType())
            .alias("position"),
            split.getItem(2).alias("allele1"),
            allele_hash(split.getItem(2), split.getItem(3)),
            # METAL outputs
            f.col("freq1").alias("effectAlleleFrequencyFromSource"),
            f.col("effect").alias("beta"),
            f.col("std_err").alias("standardError"),
            f.col("sample_count").cast(t.IntegerType()).alias("sampleSize"),
            *gentropy.common.stats.split_pvalue_column(f.col("p_value")),
        )
        .join(variants, on=["chromosome", "position", "allele_hash"])
        .withColumn(
            "beta",
            f.when(
                f.upper(f.col("allele1")) == f.col("alternateAllele"), f.col("beta")
            ).otherwise(-f.col("beta")),
        )
        .drop("allele1", "alternateAllele", "allele_hash", "lifted_coordinates")
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
    study_locus.df.coalesce(1).write.parquet("study-locus")
