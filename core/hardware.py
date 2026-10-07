"""Hardware detection (no Qt). Used by the engine and the GUI hardware picker."""


def available_devices():
    """Return a dict like {'cuda': True, 'mps': False, 'cpu': True, 'cuda_name': '...'}."""
    info = {"cuda": False, "mps": False, "cpu": True, "cuda_name": ""}
    try:
        import torch
    except Exception:
        return info
    try:
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps is not None and mps.is_available())
    except Exception:
        pass
    return info


def resolve_device(choice, info=None):
    """Turn a user choice ('auto', 'cuda', 'mps', 'cpu') into a real device name.

    Returns (device_name, note). Falls back to 'cpu' if the choice is not available.
    """
    info = info or available_devices()
    choice = (choice or "auto").lower()
    if choice == "auto":
        if info["cuda"]:
            return "cuda", "Auto picked CUDA"
        if info["mps"]:
            return "mps", "Auto picked Mac (MPS)"
        return "cpu", "Auto picked CPU"
    if choice in ("cuda", "mps") and not info[choice]:
        return "cpu", f"{choice.upper()} not found — using CPU instead"
    return choice, ""
