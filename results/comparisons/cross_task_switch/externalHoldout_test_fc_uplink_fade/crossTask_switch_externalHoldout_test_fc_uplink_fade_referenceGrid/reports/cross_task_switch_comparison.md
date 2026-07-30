# Cross-Task Switch Comparison

This comparison uses saved per-task switch-evaluation artifacts only. It does not train models or recompute task-specific switches.

## Denominators

- `reference_grid_missing_as_zero`: evaluates every method on the common Perfect Switch reference grid. Missing native method decisions are treated as switch `0`, so this is coverage-aware.
- `strict_common_timestamp_intersection`: evaluates only timestamps where every included method has a native decision, so this compares conditional decision quality but hides coverage differences.

## Top Reference-Grid F1

```text
                task_name                     method_name       f1  precision   recall  balanced_accuracy  active_duration_seconds  coverage_pct  num_points
current_level_persistence          XGBoost Scalar Context 0.893868   1.000000 0.808102           0.904051                  11370.0     88.986580        8644
     survival_persistence       XGBoost AFT Raw Flattened 0.870425   0.811808 0.938166           0.962845                  16260.0      9.220268        8644
           autoregressive                    PatchTST Raw 0.799585   0.779352 0.820896           0.903781                  14820.0     89.171680        8644
           autoregressive                     XGBoost Raw 0.790868   0.691693 0.923241           0.949816                  18780.0     89.171680        8644
     survival_persistence     Discrete-Time TCN Grid Best 0.776748   0.642061 0.982942           0.975753                  21540.0      9.220268        8644
           autoregressive                 GRU Seq2Seq Raw 0.752508   0.618982 0.959488           0.962802                  21810.0     89.171680        8644
      long_fade_detection                  TCN Classifier 0.715484   0.557007 1.000000           0.977187                  25260.0     28.690421        8644
      long_fade_detection              XGBoost Lag+Scalar 0.714939   0.556346 1.000000           0.977125                  25290.0     28.690421        8644
           autoregressive           Chronos Zero-Shot Raw 0.707298   0.566838 0.940299           0.949538                  23340.0     89.171680        8644
      long_fade_detection Shapelet Convolution Classifier 0.697466   0.536082 0.997868           0.974163                  26190.0     28.690421        8644
```

## Top Strict-Intersection F1

```text
                task_name                 method_name       f1  precision   recall  balanced_accuracy  active_duration_seconds  coverage_pct  num_points
current_level_persistence      XGBoost Scalar Context 0.900485   1.000000 0.818985           0.909492                  11130.0      8.919482         771
     survival_persistence   XGBoost AFT Raw Flattened 0.871795   0.814176 0.938190           0.816579                  15660.0      8.919482         771
           autoregressive                 XGBoost Raw 0.865792   0.797398 0.947020           0.802126                  16140.0      8.919482         771
           autoregressive             GRU Seq2Seq Raw 0.854947   0.756803 0.982340           0.766327                  17640.0      8.919482         771
           autoregressive                PatchTST Raw 0.838992   0.832609 0.845475           0.801668                  13800.0      8.919482         771
           autoregressive                 GRU S2V Raw 0.818926   0.696594 0.993377           0.688513                  19380.0      8.919482         771
           autoregressive       Chronos Zero-Shot Raw 0.815370   0.708469 0.960265           0.698686                  18420.0      8.919482         771
     survival_persistence Discrete-Time TCN Grid Best 0.778068   0.642241 0.986755           0.601868                  20880.0      8.919482         771
      long_fade_detection              TCN Classifier 0.777682   0.636236 1.000000           0.592767                  21360.0      8.919482         771
           autoregressive    XGBoost Context Standard 0.777489   0.639601 0.991170           0.597786                  21060.0      8.919482         771
```
