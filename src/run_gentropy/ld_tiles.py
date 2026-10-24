import json
import math
from pathlib import Path

import boto3  # type: ignore[import-untyped]
import pyspark.sql.functions as f
from botocore import UNSIGNED  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from gentropy.common.session import Session

session = Session(extended_spark_conf={"spark.driver.memory": "12g"})
spark = session.spark

bucket = "pan-ukb-us-east-1"
prefix = "ld_release"
base_url = f"https://{bucket}.s3.amazonaws.com/{prefix}"


def main() -> None:
    metadata = json.loads(Path("data/UKBB.EUR.ldadj.bm/metadata.json").read_text())
    block_size = metadata["blockSize"]
    n_block_rows = math.ceil(metadata["nRows"] / block_size)

    part_files = spark.createDataFrame(
        list(zip(metadata["maybeFiltered"], metadata["partFiles"])),
        ["block", "part_file"],
    )

    study_locus = (
        spark.read.parquet("study-locus")
        .select("chromosome", "locusStart", "locusEnd")
        .distinct()
    )
    # Output of ld_index.py
    ld_index = spark.read.parquet("ld-index").select("chromosome", "position", "idx")

    # BlockMatrix.filter(idx, idx) reads every block whose row and column both
    # contain one of the locus variants
    block_row = f.floor(f.col("idx") / block_size)
    block_rows = (
        ld_index.join(f.broadcast(study_locus), on="chromosome")
        .filter(f.col("position").between(f.col("locusStart"), f.col("locusEnd")))
        .groupBy("chromosome", "locusStart", "locusEnd")
        .agg(f.collect_set(block_row).alias("block_rows"))
    )
    blocks = (
        block_rows.select(f.explode("block_rows").alias("row"), "block_rows")
        .select("row", f.explode("block_rows").alias("column"))
        # Only the upper triangle is stored
        .filter(f.col("row") <= f.col("column"))
        .select((f.col("row") + f.col("column") * n_block_rows).alias("block"))
        .distinct()
    )

    path = f.concat(f.lit("UKBB.EUR.ldadj.bm/parts/"), "part_file")
    rows = (
        blocks.join(part_files, on="block")
        .select(f.concat(f.lit(f"{base_url}/"), path).alias("url"), path.alias("path"))
        .orderBy("path")
        .collect()
    )

    # One listing of all parts is faster than one request per tile
    s3 = boto3.client(
        "s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
    )
    objects = {
        content["Key"]: content
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"{prefix}/UKBB.EUR.ldadj.bm/parts/"
        )
        for content in page["Contents"]
    }

    with Path("ld_tiles.csv").open("w") as file_handle:
        file_handle.write("url,path,size,etag\n")
        for row in rows:
            # Tiles added by an earlier run are annex symlinks, possibly without content
            if (Path("data") / row.path).exists(follow_symlinks=False):
                continue
            content = objects[f"{prefix}/{row.path}"]
            etag = content["ETag"].strip('"')
            file_handle.write(f"{row.url},{row.path},{content['Size']},{etag}\n")


if __name__ == "__main__":
    main()
