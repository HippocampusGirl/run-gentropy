from pathlib import Path

import pyspark.sql.functions as f
from gentropy.common.session import Session
from gentropy.dataset.study_locus import StudyLocus
from gentropy.method.colocalisation.coloc_pip_ecaviar import ColocPIPECaviar

session = Session(extended_spark_conf={"spark.driver.memory": "12g"})

credible_sets_path = Path("credible-sets")
colocalisation_path = Path("colocalisation")


def main() -> None:
    credible_sets = StudyLocus.from_parquet(session, credible_sets_path.as_posix())
    # find_overlaps drops credible sets without a study type, which the
    # fine-mapper does not set
    credible_sets = StudyLocus(
        _df=credible_sets.df.withColumn("studyType", f.lit("gwas")),
        _schema=StudyLocus.get_schema(),
    )

    # One row per pair of credible sets that share a variant
    colocalisation = ColocPIPECaviar.colocalise(credible_sets.find_overlaps())
    colocalisation.df.write.mode("overwrite").parquet(colocalisation_path.as_posix())


if __name__ == "__main__":
    main()
