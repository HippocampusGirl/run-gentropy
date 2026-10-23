from gentropy.common.session import Session
from gentropy.datasource.pan_ukbb_ld.ld import PanUKBBLDMatrix

session = Session(start_hail=True, extended_spark_conf={"spark.driver.memory": "12g"})
spark = session.spark


def main() -> None:
    PanUKBBLDMatrix().align_ld_index_alleles(
        variant_annotation=spark.read.parquet("data/variant/*.parquet"),
        population="EUR",
        hail_table_path="data/UKBB.EUR.ldadj.variant.b38.ht",
        hail_table_output="ld-index",
    )


if __name__ == "__main__":
    main()
