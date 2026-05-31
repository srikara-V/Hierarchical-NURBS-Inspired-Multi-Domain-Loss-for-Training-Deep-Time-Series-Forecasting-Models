"""Ensure argparse Namespace has fields expected by all model implementations."""


def apply_config_defaults(args):
    if not hasattr(args, "embed_type"):
        args.embed_type = 0
    if not hasattr(args, "individual"):
        args.individual = False

    # Autoformer uses decoder stack; keep aligned with original Autoformer scripts.
    if args.model == "Autoformer":
        args.embed_type = getattr(args, "embed_type", 0)
        if args.d_ff == 2048:
            args.d_ff = 512

    if args.model in {"NLinear", "Linear"}:
        args.individual = getattr(args, "individual", False)

    return args
