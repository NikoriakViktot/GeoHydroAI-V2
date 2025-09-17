

def get_filtered_table_title(lulc, landform, slope, hand_range_):
    filters = []
    if hand_range_:
        filters.append("HAND")
    if lulc:
        filters.append("LULC")
    if landform:
        filters.append("Landform")
    if slope and (slope[0] > 0 or slope[1] < 45):
        filters.append("Slope")
    if not filters:
        return "All Territory"
    if len(filters) == 1 and filters[0] == "HAND":
        return "Floodplain (HAND)"
    return "Filtered selection: " + ", ".join(filters)

def format_selected_filters(lulc, landform, slope, hand_range):
    parts = []

    if lulc:
        lulc_str = ", ".join(lulc) if isinstance(lulc, list) else str(lulc)
        parts.append(f"LULC: {lulc_str}")

    if landform:
        landform_str = ", ".join(landform) if isinstance(landform, list) else str(landform)
        parts.append(f"Landform: {landform_str}")

    if slope and (slope[0] > 0 or slope[1] < 45):
        parts.append(f"Slope: {slope[0]}–{slope[1]}°")

    if hand_range:
        parts.append(f"HAND: {hand_range[0]}–{hand_range[1]} m")

    if not parts:
        return "Filters: None"
    return "Filters applied: " + ", ".join(parts)
