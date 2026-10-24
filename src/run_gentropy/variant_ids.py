import pyspark.sql.functions as f
import pyspark.sql.types as t
from gentropy.common.genomic_region import LiftOverSpark
from gentropy.common.session import Session
from pyspark.sql.column import Column

session = Session(extended_spark_conf={"spark.driver.memory": "12g"})
spark = session.spark
liftover = LiftOverSpark(chain_file="data/grch37_to_grch38.over.chain.gz")


def allele_hash(reference: Column, alternate: Column) -> Column:
    """Order-invariant 64-bit hash of an allele pair: hash(A) + hash(B) == hash(B) + hash(A)."""
    return (
        f.xxhash64(reference.cast(t.StringType()))
        + f.xxhash64(alternate.cast(t.StringType()))
    ).alias("allele_hash")


def main() -> None:
    # Output of ld_index.py
    ld_index = spark.read.parquet("ld-index")

    split = f.split("marker_name", ":", limit=4)
    chromosome = split.getItem(0)
    position = split.getItem(1).cast(t.IntegerType())
    reference_allele = split.getItem(2)
    alternate_allele = split.getItem(3)

    lifted_chromosome = f.col("lifted_coordinates").getItem(0)
    lifted_position = f.col("lifted_coordinates").getItem(1)

    # Output of marker_names.py
    marker_names = spark.read.parquet("marker-names")

    grch37 = (
        marker_names.filter(f.col("genome_build") == "GRCh37")
        .withColumn(
            "lifted_coordinates",
            f.explode(liftover.liftover_udf(chromosome, position)),
        )
        .select(
            "marker_name",
            "genome_build",
            lifted_chromosome.alias("chromosome"),
            lifted_position.cast(t.IntegerType()).alias("position"),
        )
    )

    # Markers already in GRCh38 need no liftover
    grch38 = marker_names.filter(f.col("genome_build") == "GRCh38").select(
        "marker_name",
        "genome_build",
        chromosome.alias("chromosome"),
        position.alias("position"),
    )

    data_frame = (
        grch37.unionByName(grch38)
        .select(
            "marker_name",
            "genome_build",
            f.regexp_replace("chromosome", "^chr", "").alias("chromosome"),
            "position",
            allele_hash(reference_allele, alternate_allele),
        )
        .join(
            ld_index.select(
                "chromosome",
                "position",
                allele_hash(f.col("referenceAllele"), f.col("alternateAllele")),
                "variantId",
                "referenceAllele",
                "alternateAllele",
            ),
            on=["chromosome", "position", "allele_hash"],
            how="left",
        )
        .drop("allele_hash")
        # Markers missing from the LD index get a fallback ID
        .withColumns(
            {
                "variantId": f.coalesce(
                    f.col("variantId"),
                    f.concat_ws(
                        "_",
                        "chromosome",
                        "position",
                        reference_allele,
                        alternate_allele,
                    ),
                ),
                "referenceAllele": f.coalesce(
                    f.col("referenceAllele"), reference_allele
                ),
                "alternateAllele": f.coalesce(
                    f.col("alternateAllele"), alternate_allele
                ),
            }
        )
    )

    data_frame.repartition(1).write.mode("overwrite").parquet("variant-ids")


if __name__ == "__main__":
    main()
