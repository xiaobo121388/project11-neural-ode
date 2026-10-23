//! File-level regression tests with bounded training, independent of any GUI.
use project11::{
    data::{DataConfig, Dataset, Normalizer, Split},
    evaluation,
    model::Model,
    nn::Activation,
    training::{self, TrainConfig},
};

fn small_data() -> Dataset {
    Dataset::generate(DataConfig {
        trajectories: 8,
        train_trajectories: 4,
        validation_trajectories: 2,
        steps: 40,
        ..DataConfig::default()
    })
    .unwrap()
}
#[test]
fn csv_training_checkpoint_and_evaluation_round_trip() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let data = small_data();
    data.save(&dir.join("data")).unwrap();
    let loaded = Dataset::load(&dir.join("data")).unwrap();
    assert_eq!(data.rows, loaded.rows);
    let config = TrainConfig {
        epochs: 8,
        ..TrainConfig::default()
    };
    let model = training::train(
        &loaded,
        &[2, 8, 2],
        Activation::Tanh,
        &config,
        &dir.join("model.json"),
        &dir.join("history.csv"),
    )
    .unwrap();
    model.save(&dir.join("model.json")).unwrap();
    let restored = Model::load(&dir.join("model.json")).unwrap();
    assert!(
        (restored.next([0.4, 0.2]).unwrap()[0] - model.next([0.4, 0.2]).unwrap()[0]).abs() < 1e-14
    );
    let summary = evaluation::evaluate(&loaded, &restored, None, &dir.join("results")).unwrap();
    assert_eq!(summary.test_samples, 80);
    assert_eq!(summary.one_step.len(), 4);
    assert_eq!(summary.rollouts.len(), 12);
    let (train_x, _) = loaded.matrices(Split::Train).unwrap();
    assert_eq!(
        model.input_normalization,
        Normalizer::fit(&train_x).unwrap()
    );
    let repeated = training::train(
        &loaded,
        &[2, 8, 2],
        Activation::Tanh,
        &config,
        &dir.join("repeat.json"),
        &dir.join("repeat.csv"),
    )
    .unwrap();
    assert_eq!(
        model.network.layers[0].weights,
        repeated.network.layers[0].weights
    );
}
#[test]
fn tampered_splits_and_unsupported_model_files_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let data = small_data();
    data.save(temp.path()).unwrap();
    let path = temp.path().join("transitions.csv");
    let content = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, content.replacen(",train,", ",test,", 1)).unwrap();
    assert!(Dataset::load(temp.path()).is_err());
    assert!(Model::load(&temp.path().join("missing.json")).is_err());
    let mut model = Model::from_json(include_str!("../models/main.json")).unwrap();
    model.schema_version = 99;
    assert!(model.save(&temp.path().join("bad.json")).is_err());
    assert!(
        DataConfig {
            steps: 100000,
            trajectories: 10000,
            ..DataConfig::default()
        }
        .validate()
        .is_err()
    );
}
#[test]
fn cli_help_is_available_without_opening_a_window() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_project11"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    for command in [
        "generate",
        "train",
        "evaluate",
        "benchmark",
        "reproduce",
        "gui",
    ] {
        assert!(text.contains(command));
    }
}
