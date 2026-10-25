from pathlib import Path

import pyspark.sql.functions as f
from gentropy.common.session import Session
from gentropy.config import _L2G_FEATURES_LIST
from gentropy.dataset.l2g_feature_matrix import L2GFeatureMatrix
from gentropy.dataset.l2g_prediction import L2GPrediction
from gentropy.dataset.study_locus import StudyLocus
from gentropy.l2g import LocusToGeneFeatureMatrixStep

session = Session(
    extended_spark_conf={"spark.driver.memory": "12g"},
    # Configure l2g-feature-matrix output
    write_mode="overwrite",
    output_partitions=1,
)

combined_credible_sets_path = Path("combined-credible-sets")
colocalisation_path = Path("colocalisation")
feature_matrix_path = Path("l2g-feature-matrix")
predictions_path = Path("l2g-predictions")


def main() -> None:
    LocusToGeneFeatureMatrixStep(
        session,
        features_list=list(_L2G_FEATURES_LIST),
        credible_set_path=combined_credible_sets_path.as_posix(),
        variant_index_path="data/variant",
        colocalisation_path=colocalisation_path.as_posix(),
        study_index_path="data/study",
        target_index_path="data/target",
        intervals_path="data/enhancer_to_gene",
        gene_interactions_path="data/interaction",
        feature_matrix_path=feature_matrix_path.as_posix(),
        append_null_features=True,
    )

    feature_matrix = L2GFeatureMatrix(
        _df=session.spark.read.parquet(feature_matrix_path.as_posix())
    )
    credible_sets = StudyLocus.from_parquet(
        session, combined_credible_sets_path.as_posix()
    )
    # LocusToGeneStep requires a Hugging Face token even for public models
    predictions = (
        L2GPrediction.from_credible_set(
            session,
            credible_sets,
            feature_matrix,
            model_path="opentargets/locus_to_gene_platform-2609",
            download_from_hub=True,
        )
        .filter(f.col("score") >= 0.05)
        .add_features(feature_matrix)
    )
    predictions.df.write.mode("overwrite").parquet(predictions_path.as_posix())


if __name__ == "__main__":
    main()
