# Switch Diagnostic Plot Navigation

Generated plots are organized by scope. Intra-task figures live under `results/comparisons/switch_diagnostics/<task>/<selection>/<method>/`. Cross-task figures live inside the corresponding `results/comparisons/cross_task_switch/<selection>/<comparison>/figures/` folder.

## Plot Counts

```text
     scope                 task_name                                     plot_type  num_plots
cross_task                       all         cross_task_event_timeline_all_methods         12
cross_task                       all    cross_task_event_timeline_selected_methods         12
cross_task                       all            reference_grid_core_switch_metrics          1
cross_task                       all               reference_grid_event_f1_heatmap          1
cross_task                       all       reference_grid_native_decision_coverage          1
cross_task                       all      reference_grid_precision_recall_coverage          1
cross_task                       all    reference_grid_raw_vs_postprocessed_counts          1
cross_task                       all       strict_intersection_core_switch_metrics          1
cross_task                       all strict_intersection_precision_recall_coverage          1
intra_task            autoregressive               autoregressive_forecast_example         27
intra_task            autoregressive                  autoregressive_horizon_error          9
intra_task            autoregressive           event_raw_vs_postprocessed_timeline         54
intra_task current_level_persistence          duration_absolute_error_distribution          4
intra_task current_level_persistence                    duration_predicted_vs_true          4
intra_task current_level_persistence           event_raw_vs_postprocessed_timeline         24
intra_task       long_fade_detection           event_raw_vs_postprocessed_timeline         18
intra_task       long_fade_detection             long_fade_probability_calibration          3
intra_task       long_fade_detection            long_fade_probability_distribution          3
intra_task      survival_persistence           event_raw_vs_postprocessed_timeline         12
intra_task      survival_persistence                               survival_curves          2
intra_task      survival_persistence              survival_probability_calibration          2
intra_task      survival_persistence             survival_probability_distribution          2
```
