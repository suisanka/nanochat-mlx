"""Evaluate CUDA memory-scenario checkpoints using the shared task ledger."""

from scripts.hybrid_eval import main as evaluate


def main(argv=None):
    return evaluate(argv, backend="cuda")


if __name__ == "__main__":
    raise SystemExit(main())
