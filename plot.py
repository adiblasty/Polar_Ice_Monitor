import pandas as pd
import matplotlib.pyplot as plt

# === Load the CSV generated earlier ===
csv_path = r"climate_output/melt_report.csv"
df = pd.read_csv(csv_path)

# Parse dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])  # remove rows with unknown dates
df = df.sort_values('date')

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['coverage_percent'], label='Ice Coverage %', color='blue', linewidth=2)

# Highlight danger points
danger_df = df[df['danger'] == 1]
plt.scatter(danger_df['date'], danger_df['coverage_percent'], color='red', label='Danger', zorder=5)

# Labels
plt.title('Polar Ice Coverage Over Time')
plt.xlabel('Date')
plt.ylabel('Ice Coverage (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save + show
plot_path = r"climate_output/melt_trend_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"📊 Trend plot saved to: {plot_path}")
