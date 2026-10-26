from pathlib import Path

import pyspark.sql.functions as f
from gentropy.common.session import Session
from gentropy.dataset.study_locus import StudyLocus
from gentropy.method.colocalisation.coloc_pip_ecaviar import ColocPIPECaviar

session = Session(extended_spark_conf={"spark.driver.memory": "12g"})

credible_sets_path = Path("credible-sets")
combined_credible_sets_path = Path("combined-credible-sets")
colocalisation_path = Path("colocalisation")


def main() -> None:
    gwas = (
        StudyLocus.from_parquet(session, credible_sets_path.as_posix())
        .df.withColumn("studyType", f.lit("gwas"))
        # Fine-mapping jobs that were interrupted wrote some credible sets twice
        .dropDuplicates(["studyLocusId"])
    )

    variant_ids = gwas.select(
        f.explode("locus.variantId").alias("variantId")
    ).distinct()

    opentargets = StudyLocus.from_parquet(session, "data/credible_set").df
    opentargets_variant_ids = opentargets.select(
        "studyLocusId", f.explode("locus.variantId").alias("variantId")
    )

    overlap = (
        opentargets_variant_ids.join(
            f.broadcast(variant_ids), on="variantId", how="left_semi"
        )
        .select("studyLocusId")
        .distinct()
    )
    opentargets = opentargets.join(overlap, on="studyLocusId", how="left_semi")

    gwas.unionByName(opentargets).write.mode("overwrite").parquet(
        combined_credible_sets_path.as_posix()
    )
    credible_sets = StudyLocus.from_parquet(
        session, combined_credible_sets_path.as_posix()
    )

    colocalisation = ColocPIPECaviar.colocalise(credible_sets.find_overlaps())
    colocalisation.df.write.mode("overwrite").parquet(colocalisation_path.as_posix())


if __name__ == "__main__":
    main()
