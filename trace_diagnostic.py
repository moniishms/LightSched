import pandas as pd

F = ["num_processes", "avg_arrival_time", "arrival_time_variance", "avg_burst_time",
     "burst_time_variance", "avg_priority", "arrival_rate"]
p = pd.read_csv("lightsched_real_world_demo_predictions.csv")
t = pd.read_csv("lightsched_ml_features.csv")

print("Columns:", p.columns.tolist())
print("\nPrediction vs windows with missing features:")
print(pd.crosstab(p["avg_burst_time"].isna(), p["predicted_scheduler"]))
print("\nWindow stride (s):", round(p["window_start_s"].diff().median(), 5),
      "| window length (s):", round((p["window_end_s"] - p["window_start_s"]).median(), 5))
print("\nTrace vs training (min / median / max):")
for f in F:
    print(f"{f:22s} trace {p[f].min():9.3f} {p[f].median():9.3f} {p[f].max():9.3f}"
          f" | train {t[f].min():9.3f} {t[f].median():9.3f} {t[f].max():9.3f}")