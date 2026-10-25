import logging
import multiprocessing
import os
import tempfile
import traceback
from functools import cache
from itertools import batched
from pathlib import Path
from subprocess import check_output

import pandas as pd
import pyspark.sql.functions as f
from gentropy.common.session import Session
from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus, StudyLocusQualityCheck
from gentropy.susie_finemapper import SusieFineMapperStep
from hail.backend import py4j_backend  # type: ignore[import-untyped]
from pyspark.sql import Row
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Started by the first task in each pool worker
worker_session: Session | None = None


def finemap(session: Session, row: Row) -> tuple[list[str], pd.DataFrame]:
    study_locus_id = row["studyLocusId"]

    study_index = StudyIndex(
        _df=session.spark.createDataFrame(
            [(row["studyId"], row["sampleSize"])], "studyId string, nSamples int"
        ).select(
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
        ),
        _schema=StudyIndex.get_schema(),
    )

    with tempfile.TemporaryDirectory() as directory:
        locus_log_path = Path(directory) / "log.tsv"
        result = SusieFineMapperStep.susie_finemapper_one_sl_row_gathered_boundaries(
            session=session,
            study_locus_row=row,
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
            log_output=locus_log_path.as_posix(),
        )

        if result is None:
            credible_sets: StudyLocus | None = None
            # The fine-mapper only writes the log to a file when it skips the locus
            log_data_frame: pd.DataFrame = pd.read_csv(locus_log_path, sep="\t")
        else:
            credible_sets = result["study_locus"]
            log_data_frame = result["log"]

    # The log also records the locus, so that the next run can skip it
    log_data_frame["studyLocusId"] = study_locus_id
    if credible_sets is None:
        return [], log_data_frame

    # As in SusieFineMapperStep, flag that the LD is from a reference panel.
    # The fine-mapper output has no qualityControls column yet
    credible_sets_data_frame = credible_sets.df.withColumn(
        "qualityControls",
        StudyLocus.update_quality_flag(
            f.array().cast("array<string>"),
            f.lit(True),
            StudyLocusQualityCheck.OUT_OF_SAMPLE_LD,
        ),
    )
    # JSON keeps the nested columns, and the main process reads it with the
    # StudyLocus schema
    return credible_sets_data_frame.toJSON().collect(), log_data_frame


@cache
def cpu_count() -> int:
    return int(check_output(["nproc"]).decode().strip())


def start_worker() -> Session:
    global worker_session
    try:
        session = Session(
            spark_uri="local[1]",
            extended_spark_conf={
                "spark.driver.memory": "48g",
                "spark.driver.maxResultSize": "0",
                # Use one core, and start with a 1 GiB heap that grows as needed
                "spark.driver.extraJavaOptions": "-XX:ActiveProcessorCount=1 -Xms1g",
                "spark.ui.enabled": "false",
                "spark.ui.showConsoleProgress": "false",
                "spark.sql.shuffle.partitions": "1",
            },
            start_hail=True,
            log_level="WARN",
        )
    except Exception:  # noqa: BLE001
        # Java exceptions cannot be sent as a pickle
        raise RuntimeError(traceback.format_exc()) from None
    # Disable retries in Hail
    py4j_backend.sync_retry_transient_errors = lambda function: function()
    return (worker_session := session)


def run(
    row: Row,
) -> tuple[str, tuple[list[str], pd.DataFrame] | None, str | None]:
    session = start_worker() if worker_session is None else worker_session
    study_locus_id = row["studyLocusId"]

    try:
        result = finemap(session, row)
    except Exception:  # noqa: BLE001
        return study_locus_id, None, traceback.format_exc()
    else:
        return study_locus_id, result, None


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    session = Session(
        spark_uri="local[1]",
        extended_spark_conf={"spark.driver.memory": "16g"},
        start_hail=False,
        log_level="WARN",
    )

    study_locus = StudyLocus.from_parquet(session, "study-locus")
    study_locus_ids = {
        row.studyLocusId for row in study_locus.df.select("studyLocusId").collect()
    }

    credible_sets_path = Path("credible-sets")
    log_path = Path("credible-sets-log.tsv")

    if log_path.exists():
        study_locus_ids -= set(
            pd.read_csv(log_path, sep="\t", usecols=["studyLocusId"]).studyLocusId
        )
    logger.info(f"Fine-mapping {len(study_locus_ids)} loci")

    # The pool reads rows from the iterator only as fast as the workers take them
    rows = (
        row
        for row in study_locus.df.toLocalIterator()
        if row.studyLocusId in study_locus_ids
    )

    processes = cpu_count()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Each worker should start its own JVM
    os.environ.pop("PYSPARK_GATEWAY_PORT", None)
    os.environ.pop("PYSPARK_GATEWAY_SECRET", None)

    multiprocessing_context = multiprocessing.get_context("forkserver")
    with multiprocessing_context.Pool(processes=processes) as pool:
        for batch in batched(
            tqdm(pool.imap_unordered(run, rows), total=len(study_locus_ids)),
            n=processes,
        ):
            credible_sets_json: list[str] = []
            log_data_frames: list[pd.DataFrame] = []
            for study_locus_id, result, error in batch:
                if error is not None:
                    logger.error(f"Fine-mapping failed for {study_locus_id}:\n{error}")
                if result is not None:
                    csj, ldf = result
                    credible_sets_json.extend(csj)
                    log_data_frames.append(ldf)

            if credible_sets_json:
                credible_sets = (
                    session.spark.read.schema(StudyLocus.get_schema())
                    # Raise on a field that does not fit the schema, instead of
                    # writing nulls
                    .option("mode", "FAILFAST")
                    .json(session.spark.sparkContext.parallelize(credible_sets_json, 1))
                )
                credible_sets.write.mode("append").parquet(
                    credible_sets_path.as_posix()
                )

            if log_data_frames:
                pd.concat(log_data_frames).to_csv(
                    log_path,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not log_path.exists(),
                )


if __name__ == "__main__":
    main()
