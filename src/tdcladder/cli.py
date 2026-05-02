from __future__ import annotations

import argparse
from pathlib import Path

from .build import build_all
from .contracts import load_contracts
from .import_gate import copy_upstream_artifacts, validate_project_local_sources
from .market_liquidity import build_trace_turnover
from .treasury_stock import build_monthly_outstanding_by_maturity, download_and_build_treasury_stock


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Project root; defaults to current working directory.")


def cmd_validate_sources(args: argparse.Namespace) -> int:
    df = validate_project_local_sources(args.root, upstream=args.upstream, required_only=args.required_only)
    cols = ["name", "upstream", "required", "status", "rows", "date_min", "date_max", "local_path"]
    print(df[cols].to_string(index=False))
    failed = df["status"].astype(str).str.startswith("failed") | df["status"].eq("unreadable") | df["status"].eq("missing_required")
    return int(failed.any())


def cmd_copy_upstream(args: argparse.Namespace) -> int:
    df = copy_upstream_artifacts(args.root, upstream=args.upstream, source_root=args.source_root, overwrite=args.overwrite)
    print(df.to_string(index=False))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    outputs = build_all(args.root, strict=args.strict, write_qa=not args.skip_qa)
    for name, df in outputs.items():
        print(f"{name}: {len(df):,} rows")
    return 0


def cmd_download_treasury_stock(args: argparse.Namespace) -> int:
    if args.from_raw:
        df = build_monthly_outstanding_by_maturity(args.root)
    else:
        df = download_and_build_treasury_stock(args.root)
    print(f"monthly_outstanding_by_maturity: {len(df):,} rows")
    return 0


def cmd_build_trace_turnover(args: argparse.Namespace) -> int:
    df = build_trace_turnover(args.root, source_path=args.source_path)
    print(f"turnover_by_maturity: {len(df):,} rows")
    return 0


def cmd_describe_contracts(args: argparse.Namespace) -> int:
    rows = []
    for contract in load_contracts(args.root, upstream=args.upstream):
        rows.append(
            {
                "name": contract.get("name"),
                "upstream": contract.get("upstream"),
                "required": contract.get("required"),
                "stage": contract.get("required_stage"),
                "local_path": contract.get("local_path"),
                "upstream_relative_path": contract.get("upstream_relative_path"),
            }
        )
    if not rows:
        print("No contracts found.")
        return 1
    widths = {k: max(len(str(k)), *(len(str(row.get(k, ""))) for row in rows)) for k in rows[0]}
    print(" ".join(k.ljust(widths[k]) for k in rows[0]))
    for row in rows:
        print(" ".join(str(row.get(k, "")).ljust(widths[k]) for k in rows[0]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tdcladder", description="Build a liquidity-weighted TDC / Treasury moneyness ladder.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-sources", help="Validate project-local data/raw/<upstream>/ files against source contracts.")
    _add_root(p)
    p.add_argument("--upstream", choices=["buycurve", "liqsub", "tdcest", "tdcpass", "market_liquidity", "treasury"], default=None)
    p.add_argument("--required-only", action="store_true", help="Validate only contracts marked required.")
    p.set_defaults(func=cmd_validate_sources)

    p = sub.add_parser("copy-upstream", help="Copy contracted artifacts from an explicit upstream root into data/raw/<upstream>/.")
    _add_root(p)
    p.add_argument("--upstream", required=True, choices=["buycurve", "liqsub", "tdcest", "tdcpass", "market_liquidity", "treasury"])
    p.add_argument("--source-root", required=True, type=Path, help="Path to the upstream project root supplied by the caller.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing local raw copies.")
    p.set_defaults(func=cmd_copy_upstream)

    p = sub.add_parser("build", help="Build monthly ladder, liqsub join, quarterly TDC comparison, and QA outputs.")
    _add_root(p)
    p.add_argument("--strict", action="store_true", help="Fail if optional comparison inputs are missing.")
    p.add_argument("--skip-qa", action="store_true", help="Skip QA output generation.")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("download-treasury-stock", help="Download FiscalData MSPD marketable detail and build optional outstanding-stock input.")
    _add_root(p)
    p.add_argument("--from-raw", action="store_true", help="Skip download and rebuild monthly stock input from data/raw/treasury/mspd_table_3_market.csv.")
    p.set_defaults(func=cmd_download_treasury_stock)

    p = sub.add_parser("build-trace-turnover", help="Build optional TRACE turnover input from FINRA public aggregate workbook files.")
    _add_root(p)
    p.add_argument("--source-path", required=True, type=Path, help="Directory or file containing FINRA TRACE Treasury aggregate .xlsx/.zip files.")
    p.set_defaults(func=cmd_build_trace_turnover)

    p = sub.add_parser("describe-contracts", help="Print configured source contracts.")
    _add_root(p)
    p.add_argument("--upstream", choices=["buycurve", "liqsub", "tdcest", "tdcpass", "market_liquidity", "treasury"], default=None)
    p.set_defaults(func=cmd_describe_contracts)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
