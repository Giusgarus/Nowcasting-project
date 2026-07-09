from src.tasks.current_level_persistence.utils.paths import (
    make_run_id,
    make_selection_id,
    model_dir,
    model_selection_dir,
    processed_dataset_dir,
    run_dir,
    switch_comparison_dir,
    switch_summary_dir,
)


def test_current_level_persistence_paths_are_task_separated() -> None:
    selection_id = make_selection_id("fc-uplink-fade.csv")
    run_id = make_run_id(
        delta=0.5,
        context_length=30,
        model_id="multiscale_shapelet_mlp",
        selection_id=selection_id,
    )

    dataset_path = processed_dataset_dir(
        delta=0.5,
        context_length=30,
        selection_id=selection_id,
    )
    result_path = run_dir(run_id)
    checkpoint_path = model_dir(run_id)

    assert selection_id == "externalHoldout_test_fc_uplink_fade"
    assert "current_level_persistence" in str(dataset_path)
    assert "current_level_persistence" in str(result_path)
    assert "current_level_persistence" in str(checkpoint_path)
    assert selection_id in str(result_path)
    assert "multiscale_shapelet_mlp" in str(checkpoint_path)
    assert "autoregressive" not in str(dataset_path)
    assert "autoregressive" not in str(result_path)
    assert "autoregressive" not in str(checkpoint_path)
    assert run_id == (
        "currentLevelPersistence_delta0p5_L30_multiscale_shapelet_mlp_"
        "externalHoldout_test_fc_uplink_fade"
    )


def test_current_level_switch_paths_are_task_scoped() -> None:
    selection_id = "externalHoldout_test_fc_uplink_fade"
    comparison = switch_comparison_dir(
        selection_id=selection_id,
        comparison_id="switch_eval_demo",
    )
    summary = switch_summary_dir(
        selection_id=selection_id,
        summary_id="shapelet_switch_summary",
    )

    assert "comparisons/switch_eval/current_level_persistence" in str(comparison)
    assert "comparisons/model_summary/current_level_persistence" in str(summary)
    assert "autoregressive" not in str(comparison)


def test_current_level_model_selection_path_is_task_scoped() -> None:
    path = model_selection_dir(
        selection_id="externalHoldout_test_fc_uplink_fade",
        search_id="shapelet_large_scalarContext_delta0p5_L30",
    )

    assert "comparisons/model_selection/current_level_persistence" in str(path)
    assert "externalHoldout_test_fc_uplink_fade" in str(path)
    assert "shapelet_large_scalarContext_delta0p5_L30" in str(path)
    assert "autoregressive" not in str(path)


def test_current_level_persistence_run_suffix_keeps_selection_separate() -> None:
    run_id = make_run_id(
        delta=0.5,
        context_length=30,
        model_id="multiscale_shapelet_transformer",
        selection_id="externalHoldout_test_fc_uplink_fade",
        run_suffix="debug_run",
    )

    result_path = run_dir(run_id)
    checkpoint_path = model_dir(run_id)

    assert run_id.endswith("__debug_run")
    assert "externalHoldout_test_fc_uplink_fade/currentLevelPersistence" in str(result_path)
    assert "debug_run" not in str(result_path.parent)
    assert "multiscale_shapelet_transformer" in str(checkpoint_path)
    assert "autoregressive" not in str(result_path)
    assert "autoregressive" not in str(checkpoint_path)
