def generate_report(stats: dict) -> str:
    window = stats.get("window_seconds", 10)
    total_unique = stats.get("total_unique_people", 0)
    in_frame = stats.get("currently_in_frame", 0)
    male = stats.get("male", 0)
    female = stats.get("female", 0)
    unknown = stats.get("unknown", 0)
    fps = stats.get("fps", 0.0)

    lines = []
    lines.append(
        f"Summary (last {window}s): {total_unique} unique people observed; {in_frame} currently in frame."
    )
    lines.append(
        f"Gender estimate: {male} male, {female} female, {unknown} unknown."
    )
    lines.append(
        f"System performance: ~{fps:.1f} FPS."
    )

    return " ".join(lines)