from pathlib import Path

from gentropy.common.session import Session
from gentropy.config import _L2G_FEATURES_LIST
from gentropy.l2g import LocusToGeneFeatureMatrixStep, LocusToGeneStep

session = Session(
    extended_spark_conf={"spark.driver.memory": "12g"},
    # Configure the outputs of the gentropy steps
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

    LocusToGeneStep(
        session,
        run_mode="predict",
        hyperparameters={},
        download_from_hub=False,
        train_on_full_dataset=False,
        credible_set_path=combined_credible_sets_path.as_posix(),
        feature_matrix_path=feature_matrix_path.as_posix(),
        model_path="data/locus_to_gene_platform-2609",
        # Only read from config.json when downloading from the hub. This is the
        # same list in the same order
        features_list=list(_L2G_FEATURES_LIST),
        predictions_path=predictions_path.as_posix(),
        l2g_threshold=0,
    )


if __name__ == "__main__":
    main()
