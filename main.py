"""
Renewables Risk Simulator

Research Question: How does high renewable (solar + wind) penetration
drive electricity price volatility in Spain's grid?

This tool fetches data from REData API and analyzes the relationship
between renewable energy share and electricity prices.

Commands:
  fetch   - Fetch price and generation data from REData API
  analyze - Run exploratory analysis, regression, and Monte Carlo simulation
  all     - Run both fetch and analyze in sequence
"""

import argparse
import sys
from datetime import datetime

from utils import validate_dates, print_header


def cmd_fetch(args):
    """Fetch data from REData API."""
    from data_fetch import fetch_and_save

    print_header("Renewables Risk Simulator - Data Fetching")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print()

    try:
        validate_dates(args.start_date, args.end_date)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        fetch_and_save(args.start_date, args.end_date, args.data_dir)
        print()
        print("Data fetching complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_analyze(args):
    """Run analysis on fetched data."""
    from data_fetch import load_data
    from analysis import run_full_analysis

    print_header("Renewables Risk Simulator - Analysis")
    print()

    try:
        df = load_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'fetch' command first to download data.")
        sys.exit(1)

    print()

    # Build Monte Carlo config from CLI args
    mc_config = {
        "high_renewable_threshold": args.threshold,
        "drop_mean": args.drop_mean,
        "drop_std": args.drop_std,
    }

    try:
        run_full_analysis(df, args.output_dir, mc_config=mc_config)
        print()
        print("Analysis complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_all(args):
    """Run both fetch and analyze."""
    from data_fetch import fetch_and_save
    from analysis import run_full_analysis

    print_header("Renewables Risk Simulator - Full Pipeline")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print()

    try:
        validate_dates(args.start_date, args.end_date)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build Monte Carlo config from CLI args
    mc_config = {
        "high_renewable_threshold": args.threshold,
        "drop_mean": args.drop_mean,
        "drop_std": args.drop_std,
    }

    try:
        # Fetch data
        df = fetch_and_save(args.start_date, args.end_date, args.data_dir)

        print()
        print()

        # Run analysis
        run_full_analysis(df, args.output_dir, mc_config=mc_config)

        print()
        print("Full pipeline complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Renewables Risk Simulator - Analyze renewable energy price volatility in Spain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for 2024
  python main.py fetch --start-date 2024-01-01 --end-date 2024-12-31

  # Run analysis on fetched data
  python main.py analyze

  # Run complete pipeline (fetch + analyze)
  python main.py all --start-date 2024-01-01 --end-date 2024-12-31

Data source: REData API (apidatos.ree.es) - No authentication required
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch command
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = "2024-01-01"

    fetch_parser = subparsers.add_parser("fetch", help="Fetch data from REData API")
    fetch_parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help=f"Start date YYYY-MM-DD (default: {default_start})"
    )
    fetch_parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help=f"End date YYYY-MM-DD (default: {default_end})"
    )
    fetch_parser.add_argument(
        "--data-dir",
        type=str,
        default="/data",
        help="Directory for data files (default: /data)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis on fetched data")
    analyze_parser.add_argument(
        "--data-dir",
        type=str,
        default="/data",
        help="Directory for data files (default: /data)"
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="/outputs",
        help="Directory for output plots and reports (default: /outputs)"
    )
    analyze_parser.add_argument(
        "--threshold",
        type=float,
        default=60,
        help="High renewable threshold %% for Monte Carlo (default: 60)"
    )
    analyze_parser.add_argument(
        "--drop-mean",
        type=float,
        default=20,
        help="Mean renewable drop %% for simulation (default: 20)"
    )
    analyze_parser.add_argument(
        "--drop-std",
        type=float,
        default=10,
        help="Std dev of renewable drop %% (default: 10)"
    )

    # All command (fetch + analyze)
    all_parser = subparsers.add_parser("all", help="Run fetch and analyze in sequence")
    all_parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help=f"Start date YYYY-MM-DD (default: {default_start})"
    )
    all_parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help=f"End date YYYY-MM-DD (default: {default_end})"
    )
    all_parser.add_argument(
        "--data-dir",
        type=str,
        default="/data",
        help="Directory for data files (default: /data)"
    )
    all_parser.add_argument(
        "--output-dir",
        type=str,
        default="/outputs",
        help="Directory for output plots and reports (default: /outputs)"
    )
    all_parser.add_argument(
        "--threshold",
        type=float,
        default=60,
        help="High renewable threshold %% for Monte Carlo (default: 60)"
    )
    all_parser.add_argument(
        "--drop-mean",
        type=float,
        default=20,
        help="Mean renewable drop %% for simulation (default: 20)"
    )
    all_parser.add_argument(
        "--drop-std",
        type=float,
        default=10,
        help="Std dev of renewable drop %% (default: 10)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
