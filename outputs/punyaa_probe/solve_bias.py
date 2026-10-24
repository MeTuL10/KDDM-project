"""Recover the test-period level bias of a submission from one leaderboard probe.

With d = prediction - actual, bias b = mean(d) and RMSE_0^2 = b^2 + var(d).
Shifting every prediction by c gives RMSE_c^2 = RMSE_0^2 + 2bc + c^2, so

    b = (RMSE_c^2 - RMSE_0^2 - c^2) / (2c)

is exact: one probe identifies b with no modelling assumption. Submitting the
baseline shifted by -b then scores sqrt(RMSE_0^2 - b^2).
"""
import sys


def solve(rmse_0: float, rmse_c: float, c: float) -> dict:
    b = (rmse_c**2 - rmse_0**2 - c**2) / (2 * c)
    best = (rmse_0**2 - b**2) ** 0.5
    return {"bias_pred_minus_actual": b, "apply_shift": -b,
            "predicted_best_rmse": best, "gain": rmse_0 - best}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: solve_bias.py <rmse_baseline> <rmse_probe> <probe_shift>")
    for key, value in solve(*map(float, sys.argv[1:])).items():
        print(f"{key:28s} {value:+.4f}")
