import argparse
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as f
from gentropy.common.session import Session
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus, StudyLocusQualityCheck
from gentropy.susie_finemapper import SusieFineMapperStep
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

results_path = Path("credible-sets")

# The fine-mapper needs Spark and Hail on the driver, so the loci run in threads
# that submit their Spark jobs to this session concurrently
session = Session(start_hail=True)
spark = session.spark


study_locus = StudyLocus.from_parquet(session, "study-locus").persist()
study_index = StudyIndex(
    _df=(
        study_locus.df.groupBy("studyId")
        .agg(f.max("sampleSize").alias("nSamples"))
        .select(
            "studyId",
            f.col("studyId").alias("projectId"),
            f.lit("gwas").alias("studyType"),
            "nSamples",
            f.array(
                f.struct(
                    f.lit("nfe").alias("ldPopulation"),  # EUR
                    f.lit(1.0).alias("relativeSampleSize"),
                )
            ).alias("ldPopulationStructure"),
            f.array().cast("array<string>").alias("qualityControls"),
            f.array().cast("array<string>").alias("analysisFlags"),
            f.lit(True).alias("hasSumstats"),
        )
    ),
    _schema=StudyIndex.get_schema(),
)


def finemap(study_locus_id: str) -> DataFrame:
    # Each thread gets its own pool
    spark.sparkContext.setLocalProperty(
        "spark.scheduler.pool", threading.current_thread().name
    )

    study_locus_row = study_locus.df.filter(
        f.col("studyLocusId") == study_locus_id
    ).collect()[0]
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "log.tsv"
        result = SusieFineMapperStep.susie_finemapper_one_sl_row_gathered_boundaries(
            session=session,
            study_locus_row=study_locus_row,
            study_index=study_index,
            ld_matrix_paths={
                "ukbb_annotation_path": "ld-index",
                "pan_ukbb_bm_path": "data/UKBB.{POP}.ldadj.bm",
            },
            # Do not filter out low-purity credible sets
            purity_mean_r2_threshold=0,
            purity_min_r2_threshold=0,
            # Where the Open Targets fine-mapping pipeline differs from the defaults
            sum_pips=0.95,
            ld_min_r2=0.8,
            log_output=log_path.as_posix(),
        )
        if result is None:
            credible_sets: StudyLocus | None = None
            # The fine-mapper only writes the log to a file when it skips the locus
            log_data_frame: pd.DataFrame = pd.read_csv(log_path, sep="\t")
        else:
            credible_sets = result["study_locus"]
            log_data_frame = result["log"]
    log_data_frame["studyLocusId"] = study_locus_id

    # The log also records the locus, so that the next run can skip it
    data_frame = spark.createDataFrame(log_data_frame).select(
        f.struct("*").alias("log")
    )
    if credible_sets is not None:
        # As in SusieFineMapperStep, flag that the LD is from a reference panel.
        # The fine-mapper output has no qualityControls column yet
        data_frame = credible_sets.df.withColumn(
            "qualityControls",
            StudyLocus.update_quality_flag(
                f.array().cast("array<string>"),
                f.lit(True),
                StudyLocusQualityCheck.OUT_OF_SAMPLE_LD,
            ),
        ).crossJoin(data_frame)
    return data_frame.localCheckpoint()


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Fine-map loci with SuSiE")
    argument_parser.add_argument("--num-threads", type=int, required=True)

    arguments = argument_parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    study_locus_ids_data_frame = study_locus.df.select("studyLocusId")
    if results_path.is_dir():
        # Skip the loci of earlier runs
        done = spark.read.parquet(results_path.as_posix()).select("log.studyLocusId")
        study_locus_ids_data_frame = study_locus_ids_data_frame.join(
            done, on="studyLocusId", how="left_anti"
        )
    study_locus_ids = sorted(
        row.studyLocusId for row in study_locus_ids_data_frame.collect()
    )
    logger.info(f"Fine-mapping {len(study_locus_ids)} loci")

    data_frame: DataFrame | None = None
    with ThreadPoolExecutor(max_workers=arguments.num_threads) as executor:
        futures = {
            executor.submit(finemap, study_locus_id): study_locus_id
            for study_locus_id in study_locus_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            study_locus_id = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.exception(f"Fine-mapping failed for {study_locus_id}")
            else:
                if data_frame is None:
                    data_frame = result
                else:
                    data_frame = data_frame.unionByName(
                        result, allowMissingColumns=True
                    )
            logger.info(f"Finished {index}/{len(futures)}: {study_locus_id}")

    if data_frame is not None:
        data_frame.coalesce(1).write.mode("append").parquet(results_path.as_posix())


if __name__ == "__main__":
    main()
