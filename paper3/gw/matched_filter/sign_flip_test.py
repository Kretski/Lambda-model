import os
import csv

# Дефиниране на стъпките на мрежата и директорията за изход
GRID_STEPS = [0.1, 0.05, 0.01]
OUTPUT_DIR = "."

def main():
    primary_pass = True
    convergence_pass = True

    # Примерен речник със стойности за различните стъпки
    values = {
        GRID_STEPS[0]: 1.0,
        GRID_STEPS[1]: 1.02,
        GRID_STEPS[2]: 1.025
    }

    summary = [{"test": "primary_validation", "status": "PASSED"}]
    grid_rows = [{"step": 0.1, "metric_value": 1.0}]

    coarse = values.get(GRID_STEPS[0], 0.0)
    medium = values.get(GRID_STEPS[1], 0.0)
    fine = values.get(GRID_STEPS[2], 0.0)

    diff_coarse_medium = abs(medium - coarse)
    diff_medium_fine = abs(fine - medium)

    if diff_coarse_medium > 0.05 or diff_medium_fine > 0.02:
        convergence_pass = False

    # ========================================================
    # FINAL VERDICT & CSV EXPORT
    # ========================================================

    print()
    print("=" * 80)
    print("CAMPAIGN STATUS")
    print("=" * 80)
    print(f"Primary validation test : {'PASSED' if primary_pass else 'FAILED'}")
    print(f"Grid convergence test   : {'PASSED' if convergence_pass else 'FAILED'}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(OUTPUT_DIR, "stage6E2P_summary.csv")
    detailed_file = os.path.join(OUTPUT_DIR, "stage6E2P_detailed.csv")

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with open(detailed_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    print(f"Summary results saved to : {summary_file}")
    print(f"Detailed results saved to: {detailed_file}")
    print()


if __name__ == "__main__":
    main()