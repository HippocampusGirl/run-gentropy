import tempfile

from gentropy.common.session import Session
from gentropy.datasource.pan_ukbb_ld.ld import PanUKBBLDMatrix

session = Session(start_hail=True, extended_spark_conf={"spark.driver.memory": "12g"})
spark = session.spark


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        unsorted_path = f"{temporary_directory}/ld-index"
        PanUKBBLDMatrix().align_ld_index_alleles(
            variant_annotation=spark.read.parquet("data/variant/*.parquet"),
            population="EUR",
            hail_table_path="data/UKBB.EUR.ldadj.variant.b38.ht",
            hail_table_output=unsorted_path,
        )
        # The fine-mapper reads the variants of one locus at a time. One sorted
        # file with small row groups lets Spark open one file and skip the rest
        (
            spark.read.parquet(unsorted_path)
            .orderBy("chromosome", "position")
            .coalesce(1)
            .write.option("parquet.block.size", 8 * 2**20)
            .mode("overwrite")
            .parquet("ld-index")
        )


if __name__ == "__main__":
    main()
