import argparse
import sys
from acoustisolar.client import AcoustiSolar
from acoustisolar_schemas.source import SourceClass, CrackType

def main():
    parser = argparse.ArgumentParser(
        prog="acoustisolar",
        description="AcoustiSolar Simulation & ML CLI for solar panel micro-crack detection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: simulate
    sim_parser = subparsers.add_parser("simulate", help="Run a single synthetic simulation")
    sim_parser.add_argument("--x", type=float, default=500.0, help="Source X position in mm")
    sim_parser.add_argument("--y", type=float, default=800.0, help="Source Y position in mm")
    sim_parser.add_argument("--temp", type=float, default=25.0, help="Temperature in Celsius")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset batch")
    gen_parser.add_argument("--n", type=int, default=20, help="Number of events to generate")
    gen_parser.add_argument("--output", type=str, default="data_events.csv", help="Output CSV path")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run held-out module ML benchmark")
    bench_parser.add_argument("--train-n", type=int, default=40, help="Number of training events")
    bench_parser.add_argument("--test-n", type=int, default=20, help="Number of held-out test events")

    args = parser.parse_args()

    client = AcoustiSolar()

    if args.command == "simulate":
        res = client.simulate_event(x_mm=args.x, y_mm=args.y, temperature_c=args.temp)
        print("=== AcoustiSolar Simulation Completed ===")
        print(f"Triggered: {res['triggered']} (Channel {res['trigger_channel']})")
        if res["localization"]:
            loc = res["localization"]
            print(f"Estimated Position: ({loc['x_mm']:.1f} mm, {loc['y_mm']:.1f} mm)")
            print(f"Uncertainty: {loc['uncertainty_semi_major_mm']:.1f} mm major axis")
            
    elif args.command == "generate":
        print(f"Generating {args.n} synthetic events...")
        _, df = client.generate_synthetic_dataset(n_events=args.n)
        df.drop(columns=["feature_vector"]).to_csv(args.output, index=False)
        print(f"Saved dataset metadata index to {args.output}")

    elif args.command == "benchmark":
        print(f"Running held-out module benchmark (Train: {args.train_n}, Test: {args.test_n})...")
        metrics = client.evaluate_held_out(n_train=args.train_n, n_test=args.test_n)
        print("=== Benchmark Results on Unseen Module Configuration ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

