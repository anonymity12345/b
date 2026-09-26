#!/usr/bin/env python3
"""Print the validated YAML execution topology for shell launchers."""

from __future__ import annotations

import argparse
import os

from ex_omni.deployment import (
    build_split_deployment_plan,
    print_deployment_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--address")
    parser.add_argument("--authkey")
    parser.add_argument(
        "--format",
        choices=("legacy", "shell", "json"),
        default="legacy",
    )
    args = parser.parse_args()
    plan = build_split_deployment_plan(
        args.config,
        address=args.address,
        authkey=args.authkey,
        environment=os.environ if args.format != "legacy" else None,
    )
    print_deployment_plan(plan, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
