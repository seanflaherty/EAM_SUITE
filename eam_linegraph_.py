"""linegrapgh module"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# Configuration Constants
FONT_CONFIG = {"title": {"size": "24", "weight": "bold"}, "tick": {"weight": "bold"}}

PREFIX_GROUPS = {
    "Group 1 (Green)": {
        "prefixes": ["YL", "YT"],
        "base_hex": "#06b050",
        "marker": "o",
        "hatch": None,
    },
    "Group 2 (Orange)": {
        "prefixes": ["T5", "F6"],
        "base_hex": "#ff9000",
        "marker": "s",
        "hatch": "//",
    },
    "Group 3 (Yellow)": {
        "prefixes": [],
        "base_hex": "#ffff00",
        "marker": "^",
        "hatch": "\\\\",
    },
    "Group 4 (Pink)": {
        "prefixes": ["6K", "JC"],
        "base_hex": "#ff0467",
        "marker": "*",
        "hatch": ".....",
    },
    "Uncategorized": {
        "prefixes": [],
        "base_hex": "#121930",
        "marker": ".",
        "hatch": None,
    },
}

VALID_MARKERS = {
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
}
SHORT_PERIOD_THRESHOLD_DAYS = 31
INTERMEDIATE_PERIOD_THRESHOLD_DAYS = 90
ALPHA_STEP = 0.4


# Modified to apply style conditionally
def apply_plot_style(bw_mode=False):
    """Apply appropriate plot style based on mode"""
    if bw_mode:
        plt.style.use("default")  # Use light background for BW mode
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
            }
        )
    else:
        plt.style.use("dark_background")


class DateProcessor:
    """Handles all date-related operations."""

    @staticmethod
    def is_short_period(start_date, end_date):
        """Returns SHORT_PERIOD_THRESHOLD_DAYS."""
        return (end_date - start_date).days <= SHORT_PERIOD_THRESHOLD_DAYS

    @staticmethod
    def is_intermediate_period(start_date, end_date):
        """Returns INTERMEDIATE_PERIOD_THRESHOLD_DAYS."""
        days = (end_date - start_date).days
        return SHORT_PERIOD_THRESHOLD_DAYS < days <= INTERMEDIATE_PERIOD_THRESHOLD_DAYS

    @staticmethod
    def parse_compact_date(date_string):
        """Parse compact date formats: '24', '2401', '240115'"""
        if len(date_string) == 2:
            return 2000 + int(date_string), None, None
        elif len(date_string) == 4:
            return 2000 + int(date_string[:2]), int(date_string[2:]), None
        elif len(date_string) == 6:
            return (
                2000 + int(date_string[:2]),
                int(date_string[2:4]),
                int(date_string[4:]),
            )
        else:
            raise ValueError(f"Invalid date format: {date_string}")

    @staticmethod
    def format_date_label(date_obj, label_type="chart", bw_mode=False):
        """Format date for different chart types with BW mode support"""
        text_color = "black" if bw_mode else "white"

        if label_type == "yaxis":
            day_abbr = date_obj.strftime("%a").upper()
            date_formatted = date_obj.strftime("%y%m%d")
            return Text(
                text=f"{day_abbr} {date_formatted}", fontweight="bold", color=text_color
            )
        else:  # chart
            day_abbr = date_obj.strftime("%a").upper()
            date_formatted = date_obj.strftime("%m%d")
            return f"{day_abbr}\n{date_formatted}"

    @staticmethod
    def get_month_boundaries(start_date, end_date):
        """Get the first day of each month within the date range"""
        current_date = pd.to_datetime(f"{start_date.year}-{start_date.month}-01").date()
        boundaries = []

        while current_date <= end_date:
            if current_date >= start_date:
                boundaries.append(current_date)
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        return boundaries

    @staticmethod
    def get_week_boundaries(start_date, end_date):
        """Get Monday of each week within the date range with ISO week numbers"""
        current_date = start_date - pd.Timedelta(days=start_date.weekday())
        boundaries = []

        while current_date <= end_date:
            if current_date >= start_date:
                iso_year, iso_week, _ = current_date.isocalendar()
                month_abbr = current_date.strftime("%b").upper()
                boundaries.append((current_date, f"W{iso_week:02d}\n{month_abbr}"))
            current_date += pd.Timedelta(days=7)

        return boundaries


class DataProcessor:
    """Handles data loading, filtering, and preprocessing"""

    @staticmethod
    def process_datetime_columns(df_input):
        """Extract and standardize datetime processing with robust error handling"""
        df_processed = df_input.copy()
        original_count = len(df_processed)

        # Convert Date to string format if it's not already
        if df_processed["Date"].dtype == "datetime64[ns]":
            df_processed["Date_str"] = df_processed["Date"].dt.strftime("%Y.%m.%d")
        else:
            df_processed["Date_str"] = df_processed["Date"].astype(str)

        # Convert Time to string format with improved error handling
        def safe_time_format(time_val):
            """Safely format time values to HH:MM string format"""
            if pd.isna(time_val):
                return ""

            # If it's already a string, return as-is (assuming it's in correct format)
            if isinstance(time_val, str):
                return str(time_val)

            # If it has strftime method (datetime-like object)
            if hasattr(time_val, "strftime"):
                try:
                    return time_val.strftime("%H:%M")
                except Exception:
                    return str(time_val)

            # For other types, convert to string
            return str(time_val)

        time_column = df_processed["Time"]
        if hasattr(time_column, "dt"):
            # Pandas datetime series
            df_processed["Time_str"] = time_column.dt.strftime("%H:%M")
        else:
            # Apply safe formatting for mixed or string types
            df_processed["Time_str"] = time_column.apply(safe_time_format)

        # Combine date and time strings, then convert to datetime with error handling
        df_processed["DateTime_combined"] = (
            df_processed["Date_str"] + " " + df_processed["Time_str"]
        )

        # Function to safely parse datetime
        def safe_datetime_parse(datetime_str):
            try:
                # First try the expected format
                return pd.to_datetime(datetime_str, format="%Y.%m.%d %H:%M")
            except (ValueError, TypeError):
                try:
                    # Try without microseconds
                    return pd.to_datetime(datetime_str, format="%Y.%m.%d %H:%M:%S")
                except (ValueError, TypeError):
                    try:
                        # Try pandas' flexible parsing
                        return pd.to_datetime(datetime_str, errors="coerce")
                    except Exception:
                        return pd.NaT

        # Apply safe parsing
        df_processed["Datetime"] = df_processed["DateTime_combined"].apply(
            safe_datetime_parse
        )

        # Remove rows with invalid datetime entries
        invalid_datetime_mask = df_processed["Datetime"].isna()
        invalid_count = invalid_datetime_mask.sum()

        if invalid_count > 0:
            print(
                f"""Warning: Found {invalid_count}
                  rows with invalid DATE/UTC entries - these will be skipped."""
            )
            df_processed = df_processed[~invalid_datetime_mask].copy()

        # Only proceed with datetime operations on valid entries
        if len(df_processed) > 0:
            df_processed["Date"] = df_processed["Datetime"].dt.date
            df_processed["Time_minutes"] = (
                df_processed["Datetime"].dt.hour * 60
                + df_processed["Datetime"].dt.minute
            )
            df_processed["DateOnly"] = df_processed["Datetime"].dt.date
            df_processed["Year"] = df_processed["Datetime"].dt.year
            df_processed["Month"] = df_processed["Datetime"].dt.month
        else:
            print("Error: No valid datetime entries found in the data.")
            # Return empty dataframe with required columns
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Time",
                    "Callsign",
                    "Prefix",
                    "Q",
                    "Message",
                    "Datetime",
                    "Time_minutes",
                    "DateOnly",
                    "Year",
                    "Month",
                ]
            )

        # Clean up temporary columns
        df_processed = df_processed.drop(
            ["Date_str", "Time_str", "DateTime_combined"], axis=1, errors="ignore"
        )

        final_count = len(df_processed)
        if invalid_count > 0:
            print(
                f"""Successfully processed {final_count}
                  rows (skipped {invalid_count} invalid entries)."""
            )

        return df_processed

    @staticmethod
    def apply_filtering(
        df_input,
        exclude_percent=True,
        exclude_backslash_entries=False,
        exclude_duplicate_entries=False,
    ):
        """Apply all filtering operations in one pass"""
        original_count = len(df_input)
        df_filtered = df_input.copy()

        # Filter out % messages by default
        if exclude_percent:
            prefix_mask = (
                ~df_filtered["Prefix"].astype(str).str.contains("%", na=False)
                & (df_filtered["Prefix"].astype(str) != "_")
                & (df_filtered["Prefix"].astype(str) != "__")
            )
            df_filtered = df_filtered[prefix_mask].copy()

        # Filter backslashes
        if exclude_backslash_entries:
            backslash_mask = (
                ~df_filtered["Q"].astype(str).str.contains("\\\\", na=False)
            )
            df_filtered = df_filtered[backslash_mask].copy()

        # Filter duplicates
        if exclude_duplicate_entries:
            df_filtered["CallsignMessage"] = (
                df_filtered["Callsign"].astype(str)
                + "|"
                + df_filtered["Message"].astype(str)
            )
            df_filtered["DuplicateKey"] = (
                df_filtered["DateOnly"].astype(str)
                + "|"
                + df_filtered["CallsignMessage"]
            )
            duplicate_mask = df_filtered.duplicated(
                subset=["DuplicateKey"], keep="first"
            )
            df_filtered = df_filtered[~duplicate_mask].copy()
            df_filtered = df_filtered.drop(
                ["CallsignMessage", "DuplicateKey"], axis=1, errors="ignore"
            )

        print(f"\nFiltered from {original_count} lines to {len(df_filtered)} lines.")
        return df_filtered


class StyleProcessor:
    """Handles styling and grouping operations"""

    @staticmethod
    def get_sorted_prefixes_by_recency(all_prefixes, df_data):
        """Helper function to sort prefixes by most recent appearance"""
        if not df_data.empty and all_prefixes:
            prefix_dates = {}
            for prefix in all_prefixes:
                prefix_data = df_data[df_data["Prefix"] == prefix]
                if not prefix_data.empty:
                    prefix_dates[prefix] = prefix_data["Datetime"].max()
                else:
                    prefix_dates[prefix] = pd.to_datetime("1900-01-01")
            return sorted(all_prefixes, key=lambda x: prefix_dates[x], reverse=True)
        return list(all_prefixes)

    @staticmethod
    def setup_prefix_grouping(df_data, is_short_period, is_intermediate_period=False):
        """Setup prefix-to-group mapping and alpha values"""
        all_prefixes = set(df_data["Prefix"].unique()) if not df_data.empty else set()

        group_to_prefix = {}
        prefix_color = {}
        prefix_alpha_timeline = {}
        prefix_alpha_bar = {}

        for group_name, group_info in PREFIX_GROUPS.items():
            if group_name == "Uncategorized":
                # For uncategorized, add any prefix not already assigned to a group
                candidate_prefixes = [
                    p for p in all_prefixes if p not in group_to_prefix
                ]
                if candidate_prefixes:
                    group_prefixes = StyleProcessor.get_sorted_prefixes_by_recency(
                        candidate_prefixes, df_data
                    )
                    for prefix in group_prefixes:
                        group_to_prefix[prefix] = group_name
                        # Dynamically add to the prefixes list for uncategorized
                        if prefix not in group_info["prefixes"]:
                            group_info["prefixes"].append(prefix)
            else:
                # For defined groups, get candidates and sort by chronological appearance
                candidate_prefixes = [
                    p for p in group_info["prefixes"] if p in all_prefixes
                ]
                if candidate_prefixes:
                    group_prefixes = StyleProcessor.get_sorted_prefixes_by_recency(
                        candidate_prefixes, df_data
                    )
                    for prefix in group_prefixes:
                        group_to_prefix[prefix] = group_name

            # Apply color and alpha logic for all groups with prefixes
            group_prefixes_final = [
                p for p in all_prefixes if group_to_prefix.get(p) == group_name
            ]
            if not group_prefixes_final:
                continue

            # Sort final prefixes for consistent alpha assignment
            group_prefixes_final = StyleProcessor.get_sorted_prefixes_by_recency(
                group_prefixes_final, df_data
            )

            if group_name == "Uncategorized":
                for prefix in group_prefixes_final:
                    prefix_color[prefix] = group_info["base_hex"]
                    prefix_alpha_timeline[prefix] = 0.799
                    prefix_alpha_bar[prefix] = 1.0
            else:
                # Progressive alpha based on recency
                for idx, prefix in enumerate(group_prefixes_final):
                    prefix_color[prefix] = group_info["base_hex"]

                    if is_short_period or is_intermediate_period:
                        alpha_timeline = 0.799 - idx * ALPHA_STEP
                        prefix_alpha_timeline[prefix] = max(0.1, alpha_timeline)

                        alpha_bar = 1.0 - idx * ALPHA_STEP
                        prefix_alpha_bar[prefix] = max(0.1, alpha_bar)
                    else:
                        prefix_alpha_timeline[prefix] = 0.8
                        prefix_alpha_bar[prefix] = 1.0

        return group_to_prefix, prefix_color, prefix_alpha_timeline, prefix_alpha_bar

    @staticmethod
    def create_legend_handles(
        all_prefixes,
        group_to_prefix_mapping,
        daily_counts=None,
        bw_mode=False,
        is_bar_chart=False,
        df_data=None,
    ):
        """Create legend handles with consistent formatting"""
        legend_handles = []

        for group_name, info in PREFIX_GROUPS.items():
            # Get prefixes that exist in the data for this group
            candidate_prefixes = [p for p in info["prefixes"] if p in all_prefixes]

            # Sort by most recent appearance for display (most recent last for legend)
            if df_data is not None and candidate_prefixes:
                group_prefixes_for_legend = (
                    StyleProcessor.get_sorted_prefixes_by_recency(
                        candidate_prefixes, df_data
                    )
                )
                group_prefixes_for_legend.reverse()  # Most recent last for legend display
            else:
                group_prefixes_for_legend = candidate_prefixes

            # Filter prefixes for legend based on data availability
            if is_bar_chart and daily_counts is not None:
                group_prefixes_with_data = [
                    p
                    for p in group_prefixes_for_legend
                    if p in daily_counts.columns
                    and daily_counts[p].sum() > 0
                    and "%" not in str(p)
                ]
            else:
                group_prefixes_with_data = [
                    p for p in group_prefixes_for_legend if "%" not in str(p)
                ]

            if not group_prefixes_with_data:
                continue

            # Format group title
            if group_name == "Uncategorized":
                group_title = "Uncategorized"
            else:
                group_num = group_name.split()[1]
                group_title = f"Group {group_num}"

            # Wrap prefix text
            wrapped_lines = []
            for i in range(0, len(group_prefixes_with_data), 4):
                line_text = ", ".join(group_prefixes_with_data[i : i + 4])
                if i + 4 < len(group_prefixes_with_data):
                    line_text += ","
                wrapped_lines.append(line_text)
            prefix_text = "\n".join(wrapped_lines)
            label_text = f"{group_title}\n{prefix_text}"

            # Create legend handle with BW mode considerations
            legend_marker = info["marker"] if bw_mode else "s"
            legend_edge_color = "black" if bw_mode else None
            legend_line_width = 0.5 if bw_mode else 0

            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker=legend_marker,
                    linestyle="",
                    color=info["base_hex"],
                    label=label_text,
                    markeredgecolor=legend_edge_color,
                    markeredgewidth=legend_line_width,
                    markersize=15,
                )
            )

        return legend_handles

    @staticmethod
    def apply_font_styling(plot_obj, bw_mode=False):
        """Apply consistent font styling to plot elements"""
        for label in plot_obj.gca().get_xticklabels():
            label.set_fontweight(FONT_CONFIG["tick"]["weight"])
        for label in plot_obj.gca().get_yticklabels():
            label.set_fontweight(FONT_CONFIG["tick"]["weight"])


class PlotCreator:
    """Handles plot creation"""

    @staticmethod
    def create_timeline_plot(
        df_data,
        dates_sorted,
        date_to_y,
        group_to_prefix_mapping,
        prefix_color_mapping,
        prefix_alpha_timeline_mapping,
        is_short_period,
        bw_mode,
        period_start,
        period_end,
        include_percent,
        secret_marker=None,
        is_intermediate_period=False,
    ):
        """Create the timeline scatter plot"""
        apply_plot_style(bw_mode)
        plt.figure(figsize=(16, 10))

        all_prefixes = set(df_data["Prefix"].unique()) if not df_data.empty else set()

        # Plot data points with optimized marker selection
        for group_name, group_info in PREFIX_GROUPS.items():
            # Get prefixes for this group
            candidate_prefixes = [
                p for p in group_info["prefixes"] if p in all_prefixes
            ]
            if candidate_prefixes:
                group_prefixes = StyleProcessor.get_sorted_prefixes_by_recency(
                    candidate_prefixes, df_data
                )
            else:
                group_prefixes = []

            for prefix in group_prefixes:
                subset_data = df_data[df_data["Prefix"] == prefix]
                if subset_data.empty:
                    continue

                # Determine marker shape
                if secret_marker is not None:
                    marker_shape = secret_marker
                else:
                    if bw_mode:
                        marker_shape = group_info["marker"]
                    elif not is_short_period and not is_intermediate_period:
                        marker_shape = "_"
                    else:
                        marker_shape = "d"

                # Plot with appropriate styling
                if bw_mode:
                    plt.scatter(
                        subset_data["Time_minutes"],
                        subset_data["y"],
                        color=prefix_color_mapping[prefix],
                        s=150,
                        alpha=prefix_alpha_timeline_mapping[prefix],
                        marker=marker_shape,
                        edgecolors="black",
                        linewidths=0.6,
                    )
                else:
                    plt.scatter(
                        subset_data["Time_minutes"],
                        subset_data["y"],
                        color=prefix_color_mapping[prefix],
                        s=50,
                        alpha=prefix_alpha_timeline_mapping[prefix],
                        marker=marker_shape,
                    )

        PlotCreator._setup_timeline_styling(
            dates_sorted,
            date_to_y,
            is_short_period,
            period_start,
            period_end,
            include_percent,
            bw_mode,
            is_intermediate_period,
        )

        # Add legend with BW mode styling
        legend_handles = StyleProcessor.create_legend_handles(
            all_prefixes, group_to_prefix_mapping, bw_mode=bw_mode, df_data=df_data
        )
        legend_face_color = "white" if bw_mode else "black"
        legend_edge_color = "black" if bw_mode else "white"
        legend_label_color = "black" if bw_mode else "white"

        plt.legend(
            handles=legend_handles,
            title="Groups",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            facecolor=legend_face_color,
            edgecolor=legend_edge_color,
            labelcolor=legend_label_color,
            title_fontsize=12,
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_bar_chart(
        df_data,
        complete_dates,
        group_to_prefix_mapping,
        prefix_color_mapping,
        prefix_alpha_bar_mapping,
        is_short_period,
        bw_mode,
        period_start,
        period_end,
        include_percent,
        is_intermediate_period=False,
    ):
        """Create the stacked bar chart"""
        apply_plot_style(bw_mode)

        # Prepare data efficiently
        if len(df_data) > 0:
            daily_counts_raw = (
                df_data.groupby(["Date", "Prefix"]).size().unstack(fill_value=0)
            )
            daily_counts = daily_counts_raw.reindex(complete_dates, fill_value=0)
            all_prefixes_in_data = set(df_data["Prefix"].unique())
            for prefix in all_prefixes_in_data:
                if prefix not in daily_counts.columns:
                    daily_counts[prefix] = 0
        else:
            daily_counts = pd.DataFrame(index=complete_dates)
            all_prefixes_in_data = set()

        plt.figure(figsize=(16, 10))

        # Draw bars in chronological order (most recent first within each group)
        bottom_values = [0] * len(daily_counts)
        bar_width = 1
        use_edges = is_short_period or is_intermediate_period

        # Process groups in the order they appear in PREFIX_GROUPS
        for group_name, group_info in PREFIX_GROUPS.items():
            candidate_prefixes = [
                p for p in group_info["prefixes"] if p in all_prefixes_in_data
            ]

            if candidate_prefixes:
                group_prefixes = StyleProcessor.get_sorted_prefixes_by_recency(
                    candidate_prefixes, df_data
                )
            else:
                group_prefixes = []

            for prefix in group_prefixes:
                if prefix not in daily_counts.columns:
                    continue

                values = daily_counts[prefix].values

                if values.sum() == 0:
                    continue

                # Prepare styling with BW mode considerations
                hatch_pattern = group_info["hatch"] if bw_mode else None
                face_color = mcolors.to_rgba(
                    prefix_color_mapping[prefix], alpha=prefix_alpha_bar_mapping[prefix]
                )

                bar_kwargs = {
                    "color": face_color,
                    "width": bar_width,
                    "hatch": hatch_pattern,
                }

                if use_edges:
                    edge_color = "black" if bw_mode else "black"
                    bar_kwargs.update({"edgecolor": edge_color, "linewidth": 1.5})

                plt.bar(
                    range(len(daily_counts)), values, bottom=bottom_values, **bar_kwargs
                )
                bottom_values = [b + v for b, v in zip(bottom_values, values)]

        PlotCreator._setup_bar_chart_styling(
            daily_counts,
            is_short_period,
            period_start,
            period_end,
            include_percent,
            bw_mode,
            complete_dates,
            is_intermediate_period,
        )

        # Add legend with BW mode styling
        legend_handles = StyleProcessor.create_legend_handles(
            all_prefixes_in_data,
            group_to_prefix_mapping,
            daily_counts,
            bw_mode,
            is_bar_chart=True,
            df_data=df_data,
        )
        legend_face_color = "white" if bw_mode else "black"
        legend_edge_color = "black" if bw_mode else "white"
        legend_label_color = "black" if bw_mode else "white"

        plt.legend(
            handles=legend_handles,
            title="Groups",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            facecolor=legend_face_color,
            edgecolor=legend_edge_color,
            labelcolor=legend_label_color,
            title_fontsize=12,
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _setup_timeline_styling(
        dates_sorted,
        date_to_y,
        is_short_period,
        period_start,
        period_end,
        include_percent,
        bw_mode,
        is_intermediate_period=False,
    ):
        """Setup styling for timeline plot"""
        text_color = "black" if bw_mode else "white"
        grid_color = "gray" if bw_mode else "grey"

        plt.xlabel("Time (UTC)", color=text_color, fontsize=12, fontweight="bold")
        plt.ylabel("Date", color=text_color, fontsize=12, fontweight="bold")

        title_text = "HFGCS Daily Traffic Timeline"
        if include_percent:
            title_text += " (% Mode)"
        if bw_mode:
            title_text += " (BW Mode)"

        plt.title(title_text, color=text_color, fontsize=24, fontweight="bold")

        # Smart y-axis labeling
        if is_intermediate_period:
            # Week-based labeling for intermediate periods
            week_boundaries = DateProcessor.get_week_boundaries(
                period_start, period_end
            )
            y_positions = []
            y_labels = []

            for boundary_date, week_label in week_boundaries:
                if boundary_date in date_to_y:
                    y_positions.append(date_to_y[boundary_date])
                    y_labels.append(
                        Text(text=week_label, fontweight="bold", color=text_color)
                    )

            plt.yticks(y_positions, y_labels)

            for pos in y_positions:
                plt.axhline(y=pos, color=grid_color, linestyle=":", alpha=0.3)
        elif not is_short_period:
            month_boundaries = DateProcessor.get_month_boundaries(
                period_start, period_end
            )
            y_positions = []
            y_labels = []

            for boundary_date in month_boundaries:
                if boundary_date in date_to_y:
                    y_positions.append(date_to_y[boundary_date])
                    month_label = boundary_date.strftime("%b %Y").upper()
                    y_labels.append(
                        Text(text=month_label, fontweight="bold", color=text_color)
                    )

            plt.yticks(y_positions, y_labels)

            for pos in y_positions:
                plt.axhline(y=pos, color=grid_color, linestyle=":", alpha=0.3)
        else:
            y_tick_labels = [
                DateProcessor.format_date_label(d, "yaxis", bw_mode)
                for d in dates_sorted[::-1]
            ]
            plt.yticks(range(len(dates_sorted)), y_tick_labels)

            reversed_dates_timeline = dates_sorted[::-1]
            for i, date_val in enumerate(reversed_dates_timeline):
                if date_val.weekday() >= 5:
                    plt.gca().get_yticklabels()[i].set_color(text_color)
                    plt.gca().get_yticklabels()[i].set_alpha(0.8)

        plt.xlim(-30, 1470)
        plt.xticks(
            ticks=range(0, 1441, 60),
            labels=[f"{h:02d}:00" for h in range(0, 25, 1)],
            color=text_color,
        )

        if len(dates_sorted) > 0:
            y_padding = 1.0
            plt.ylim(-y_padding, len(dates_sorted) - 1 + y_padding)
        else:
            plt.ylim(-0.5, 0.5)

        StyleProcessor.apply_font_styling(plt, bw_mode)
        plt.gca().tick_params(axis="both", which="both", length=0)
        plt.grid(True, which="both", linestyle="--", alpha=0.5, color=grid_color)

    @staticmethod
    def _setup_bar_chart_styling(
        daily_counts,
        is_short_period,
        period_start,
        period_end,
        include_percent,
        bw_mode,
        complete_dates,
        is_intermediate_period=False,
    ):
        """Setup styling for bar chart"""
        text_color = "black" if bw_mode else "white"
        grid_color = "gray" if bw_mode else "grey"

        plt.xlabel("Date", color=text_color, fontsize=12, fontweight="bold")
        plt.ylabel(
            "Number of Messages", color=text_color, fontsize=12, fontweight="bold"
        )

        if len(daily_counts) > 0:
            max_total = daily_counts.sum(axis=1).max()
            plt.ylim(0, max_total * 1.1)

        title_text = "HFGCS Messages per Day by Prefix"
        if include_percent:
            title_text += " (% Mode)"
        if bw_mode:
            title_text += " (BW Mode)"

        plt.title(title_text, color=text_color, fontsize=24, fontweight="bold")

        # Smart x-axis labeling
        num_bars = len(daily_counts)
        plt.xlim(-0.5, num_bars - 0.5)

        if is_intermediate_period:
            # Week-based labeling for intermediate periods
            week_boundaries = DateProcessor.get_week_boundaries(
                period_start, period_end
            )
            x_positions = []
            x_labels = []

            for boundary_date, week_label in week_boundaries:
                if boundary_date in complete_dates:
                    x_position = complete_dates.index(boundary_date)
                    x_positions.append(x_position)
                    x_labels.append(week_label)

            plt.xticks(x_positions, x_labels, color=text_color)

            for pos in x_positions:
                plt.axvline(x=pos, color=grid_color, linestyle=":", alpha=0.3)
        elif not is_short_period:
            month_boundaries = DateProcessor.get_month_boundaries(
                period_start, period_end
            )
            x_positions = []
            x_labels = []

            for boundary_date in month_boundaries:
                if boundary_date in complete_dates:
                    x_position = complete_dates.index(boundary_date)
                    x_positions.append(x_position)
                    month_label = boundary_date.strftime("%b\n%Y").upper()
                    x_labels.append(month_label)

            plt.xticks(x_positions, x_labels, color=text_color)

            for pos in x_positions:
                plt.axvline(x=pos, color=grid_color, linestyle=":", alpha=0.3)
        else:
            date_labels_chart = [
                DateProcessor.format_date_label(date_val, "chart", bw_mode)
                for date_val in daily_counts.index
            ]
            plt.xticks(range(num_bars), date_labels_chart, color=text_color)

            for i, date_val in enumerate(daily_counts.index):
                if date_val.weekday() >= 5:
                    plt.gca().get_xticklabels()[i].set_color(text_color)
                    plt.gca().get_xticklabels()[i].set_alpha(0.8)

        StyleProcessor.apply_font_styling(plt, bw_mode)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5, color=grid_color)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.gca().tick_params(axis="both", which="both", length=0)


class UserInterface:
    """Handles user input and interaction"""

    @staticmethod
    def parse_user_input(user_input_str):
        """Parse user input for special modes and markers"""
        secret_marker_found = None
        initial_date_found = None

        # Check for date + marker combinations (e.g., "2507 x")
        choice_parts = user_input_str.split()
        if len(choice_parts) == 2:
            potential_date, potential_marker = choice_parts
            if potential_marker in VALID_MARKERS:
                try:
                    DateProcessor.parse_compact_date(potential_date)
                    secret_marker_found = potential_marker
                    initial_date_found = potential_date
                    return (
                        "",
                        False,
                        False,
                        secret_marker_found,
                        initial_date_found,
                        False,
                    )
                except ValueError:
                    pass

        # Check for marker with exclamation mark only (e.g., "d!")
        if (
            len(user_input_str) == 2
            and user_input_str.endswith("!")
            and user_input_str[0] in VALID_MARKERS
        ):
            secret_marker_found = user_input_str[0]
            return "", False, False, secret_marker_found, None, False

        # Parse special modes
        include_percent_flag = user_input_str == "%"
        BW_MODE_FLAG = user_input_str.lower() == "bw"
        use_detailed_flag = user_input_str not in ["", "%", "bw"]

        return (
            user_input_str,
            include_percent_flag,
            BW_MODE_FLAG,
            secret_marker_found,
            initial_date_found,
            use_detailed_flag,
        )


def process_date_range_input(
    user_date_input,
    main_df,
    latest_year,
    latest_month,
    include_percent_flag,
    EXCLUDE_BACKSLASHES_FLAG,
    EXCLUDE_DUPLICATES_FLAG,
    df_raw,
):
    """Process user date input and return filtered dataframe and date range info"""
    secret_marker_detected = None

    # Handle % re-inclusion in expanded mode
    if "%" in user_date_input and not include_percent_flag:
        df_with_percent = df_raw[
            ["Date", "Time", "Callsign", "Prefix", "Q", "Message"]
        ].copy()
        df_with_percent = DataProcessor.process_datetime_columns(df_with_percent)
        main_df = DataProcessor.apply_filtering(
            df_with_percent,
            exclude_percent=False,
            exclude_backslash_entries=EXCLUDE_BACKSLASHES_FLAG,
            exclude_duplicate_entries=EXCLUDE_DUPLICATES_FLAG,
        )
        include_percent_flag = True
        user_date_input = user_date_input.replace("%", "").strip()

    # Check for marker specification at the end of input
    input_parts = user_date_input.split()
    if len(input_parts) > 0:
        last_part = input_parts[-1]
        if (
            len(last_part) == 2
            and last_part.endswith("!")
            and last_part[0] in VALID_MARKERS
        ):
            secret_marker_detected = last_part[0]
            print(f"Marker '{secret_marker_detected}' will be used for timeline plots.")
            user_date_input = " ".join(input_parts[:-1]).strip()
            input_parts = user_date_input.split() if user_date_input else []
        elif len(last_part) == 1 and last_part in VALID_MARKERS:
            secret_marker_detected = last_part
            print(f"Marker '{secret_marker_detected}' will be used for timeline plots.")
            user_date_input = " ".join(input_parts[:-1]).strip()
            input_parts = user_date_input.split() if user_date_input else []

    # Parse date input
    if not user_date_input:
        df_filtered = main_df[
            (main_df["Year"] == latest_year) & (main_df["Month"] == latest_month)
        ].copy()
        period_start_date = pd.to_datetime(f"{latest_year}-{latest_month}-01").date()
        period_end_date = (
            pd.to_datetime(f"{latest_year}-{latest_month}-01") + pd.offsets.MonthEnd(0)
        ).date()
        DATE_DESCRIPTION = pd.to_datetime(f"{latest_year}-{latest_month}-01").strftime(
            "%B %Y"
        )
    else:
        input_parts = user_date_input.split()

        if len(input_parts) == 1:
            selected_year, selected_month, selected_day = (
                DateProcessor.parse_compact_date(input_parts[0])
            )

            if selected_month is None:
                df_filtered = main_df[main_df["Year"] == selected_year].copy()
                period_start_date = pd.to_datetime(f"{selected_year}-01-01").date()
                period_end_date = pd.to_datetime(f"{selected_year}-12-31").date()
                DATE_DESCRIPTION = str(selected_year)
            elif selected_day is None:
                df_filtered = main_df[
                    (main_df["Year"] == selected_year)
                    & (main_df["Month"] == selected_month)
                ].copy()
                period_start_date = pd.to_datetime(
                    f"{selected_year}-{selected_month}-01"
                ).date()
                period_end_date = (
                    pd.to_datetime(f"{selected_year}-{selected_month}-01")
                    + pd.offsets.MonthEnd(0)
                ).date()
                DATE_DESCRIPTION = pd.to_datetime(
                    f"{selected_year}-{selected_month}-01"
                ).strftime("%B %Y")
            else:
                target_date = pd.to_datetime(
                    f"{selected_year}-{selected_month:02d}-{selected_day:02d}"
                )
                df_filtered = main_df[
                    main_df["Datetime"].dt.date == target_date.date()
                ].copy()
                period_start_date = target_date.date()
                period_end_date = target_date.date()
                DATE_DESCRIPTION = target_date.strftime("%B %d, %Y")

        elif len(input_parts) == 2:
            start_year, start_month, start_day = DateProcessor.parse_compact_date(
                input_parts[0]
            )
            end_year, end_month, end_day = DateProcessor.parse_compact_date(
                input_parts[1]
            )

            if start_day is not None:
                period_start_date = pd.to_datetime(
                    f"{start_year}-{start_month:02d}-{start_day:02d}"
                ).date()
            elif start_month is not None:
                period_start_date = pd.to_datetime(
                    f"{start_year}-{start_month:02d}-01"
                ).date()
            else:
                period_start_date = pd.to_datetime(f"{start_year}-01-01").date()

            if end_day is not None:
                period_end_date = pd.to_datetime(
                    f"{end_year}-{end_month:02d}-{end_day:02d}"
                ).date()
            elif end_month is not None:
                period_end_date = (
                    pd.to_datetime(f"{end_year}-{end_month:02d}-01")
                    + pd.offsets.MonthEnd(0)
                ).date()
            else:
                period_end_date = pd.to_datetime(f"{end_year}-12-31").date()

            df_filtered = main_df[
                (main_df["Datetime"].dt.date >= period_start_date)
                & (main_df["Datetime"].dt.date <= period_end_date)
            ].copy()

            if period_start_date == period_end_date:
                DATE_DESCRIPTION = period_start_date.strftime("%B %d, %Y")
            elif period_start_date.year == period_end_date.year:
                if period_start_date.month == period_end_date.month:
                    DATE_DESCRIPTION = f"""{period_start_date.strftime('%B %d')}
                      - {period_end_date.strftime('%d, %Y')}"""
                else:
                    DATE_DESCRIPTION = f"""{period_start_date.strftime('%B %d')}
                      - {period_end_date.strftime('%B %d, %Y')}"""
            else:
                DATE_DESCRIPTION = f"""{period_start_date.strftime('%B %d, %Y')}
                  - {period_end_date.strftime('%B %d, %Y')}"""

    return (
        df_filtered,
        period_start_date,
        period_end_date,
        DATE_DESCRIPTION,
        secret_marker_detected,
        include_percent_flag,
    )


def load_csv_data(file_path):
    """Load and validate CSV data with flexible column handling"""
    try:
        # Try reading with different encodings
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        df_raw = None

        for encoding in encodings:
            try:
                df_raw = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df_raw is None:
            raise ValueError("Could not read CSV file with any supported encoding")

        print(f"Loaded {len(df_raw)} rows from {file_path}")
        print(f"Available columns: {list(df_raw.columns)}")

        # Map columns flexibly - check for common variations
        column_mapping = {}

        # Date column variations
        date_cols = ["DATE", "Date", "date", "DATE/TIME", "DateTime", "Datetime"]
        for col in date_cols:
            if col in df_raw.columns:
                column_mapping["Date"] = col
                break

        # Time column variations
        time_cols = ["UTC", "Time", "time", "UTC_TIME", "TIME"]
        for col in time_cols:
            if col in df_raw.columns:
                column_mapping["Time"] = col
                break

        # Callsign column variations
        callsign_cols = ["CALLSIGN", "Callsign", "callsign", "CALL", "Call"]
        for col in callsign_cols:
            if col in df_raw.columns:
                column_mapping["Callsign"] = col
                break

        # Prefix column variations
        prefix_cols = ["PR", "Prefix", "prefix", "PREFIX", "PFX"]
        for col in prefix_cols:
            if col in df_raw.columns:
                column_mapping["Prefix"] = col
                break

        # Q column variations
        q_cols = ["Q", "q", "QUALITY", "Quality", "QSL"]
        for col in q_cols:
            if col in df_raw.columns:
                column_mapping["Q"] = col
                break

        # Message column variations
        msg_cols = ["MESSAGE", "Message", "message", "MSG", "TEXT", "Content"]
        for col in msg_cols:
            if col in df_raw.columns:
                column_mapping["Message"] = col
                break

        # Check if we found all required columns
        required_cols = ["Date", "Time", "Callsign", "Prefix", "Q", "Message"]
        missing_cols = []

        for req_col in required_cols:
            if req_col not in column_mapping:
                missing_cols.append(req_col)

        if missing_cols:
            print(f"Warning: Could not find columns for: {missing_cols}")
            print("Available columns in CSV:")
            for i, col in enumerate(df_raw.columns):
                print(f"  {i}: {col}")

            # Let user manually map columns
            for missing_col in missing_cols:
                while True:
                    user_input = input(
                        f"Enter column name or number for {missing_col}: "
                    ).strip()
                    if user_input.isdigit():
                        col_idx = int(user_input)
                        if 0 <= col_idx < len(df_raw.columns):
                            column_mapping[missing_col] = df_raw.columns[col_idx]
                            break
                    elif user_input in df_raw.columns:
                        column_mapping[missing_col] = user_input
                        break
                    print("Invalid input. Try again.")

        # Create the standardized dataframe
        main_df = pd.DataFrame()
        for std_col, csv_col in column_mapping.items():
            main_df[std_col] = df_raw[csv_col]

        print(f"Column mapping applied:")
        for std_col, csv_col in column_mapping.items():
            print(f"  {std_col} <- {csv_col}")

        return main_df, df_raw

    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Step 1: Load the data with error handling
CSV_FILE_PATH = "SHORTWAVES.csv"

try:
    main_df, df_raw = load_csv_data(CSV_FILE_PATH)
    print(f"Successfully loaded {len(main_df)} lines from {CSV_FILE_PATH}")
except Exception as e:
    print(f"Error loading data from {CSV_FILE_PATH}: {e}")
    exit(1)

# Step 2: Process user options
print("\n" + "=" * 60)
print("PROCESSING OPTIONS")
print("=" * 60)
print("Input NULL for express mode:")
print("  • Processes most recent month in dataset")
print("  • Includes all duplicate messages")
print("  • Includes all disregards")
print("")
print("Input % or BW for special options:")
print("  • % reincludes % lines and mangled prefixes")
print("  • BW enables light background mode for better printing")
print("")
print("Input ANY for additional options:")
print("  • Choosing whether to exclude duplicates")
print("  • Choosing whether to exclude disregards")
print("  • Specifying a date range for the report")
print("=" * 60)

user_input_raw = input("Input: ")

# Parse user input
parsed_result = UserInterface.parse_user_input(user_input_raw)
PARSED_INPUT = parsed_result[0]
include_percent_flag = parsed_result[1]
BW_MODE_FLAG = parsed_result[2]
secret_marker_detected = parsed_result[3]
initial_date_detected = parsed_result[4]
use_detailed_options_flag = len(parsed_result) > 5 and parsed_result[5]

if (
    include_percent_flag
    or BW_MODE_FLAG
    or secret_marker_detected
    or initial_date_detected
):
    if secret_marker_detected:
        print(
            f"""\nSpecial marker input detected: '{secret_marker_detected}'
              will be used for timeline plots."""
        )
    if initial_date_detected:
        print(f"Date input detected: {initial_date_detected}")
    if include_percent_flag:
        print("Processing with % entries included.")
    if BW_MODE_FLAG:
        print("Light background mode enabled for better printing compatibility.")
    PARSED_INPUT = ""

if PARSED_INPUT == "":
    print(
        f"""\nExpress mode activated{' with special options'
                                      if any([include_percent_flag, BW_MODE_FLAG, secret_marker_detected, initial_date_detected])
                                        else ''}."""
    )
else:
    print("\nDetailed options mode activated.")

# Step 3: Process datetime columns
main_df = DataProcessor.process_datetime_columns(main_df)

# Check if we have any valid data after datetime processing
if len(main_df) == 0:
    print("No valid data found after processing. Exiting.")
    exit(1)

# Step 4: Apply filtering
EXCLUDE_BACKSLASHES_FLAG = False
EXCLUDE_DUPLICATES_FLAG = False

if use_detailed_options_flag:
    # Check for backslashes
    backslash_count = main_df["Q"].astype(str).str.contains("\\\\", na=False).sum()
    if backslash_count > 0:
        print(
            f"\nFound {backslash_count} lines containing '\\' in Q column out of {len(main_df)} lines."
        )
        print(f"This represents {(backslash_count/len(main_df))*100:.1f}% of the data.")
        exclude_choice = (
            input(
                "Input NULL to include these lines, or input X to exclude them.\nInput: "
            )
            .strip()
            .lower()
        )
        EXCLUDE_BACKSLASHES_FLAG = exclude_choice == "x"
        if EXCLUDE_BACKSLASHES_FLAG:
            print(f"{backslash_count} lines excluded.")

    # Check for duplicates
    main_df_temp = main_df.copy()
    main_df_temp["CallsignMessage"] = (
        main_df_temp["Callsign"].astype(str) + "|" + main_df_temp["Message"].astype(str)
    )
    main_df_temp["DuplicateKey"] = (
        main_df_temp["DateOnly"].astype(str) + "|" + main_df_temp["CallsignMessage"]
    )
    duplicate_count = main_df_temp.duplicated(
        subset=["DuplicateKey"], keep="first"
    ).sum()

    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} lines with duplicate messages.")
        print(
            f"This represents {(duplicate_count/len(main_df))*100:.1f}% of the current data."
        )
        exclude_duplicates_choice = (
            input(
                "Input NULL to include these lines, or input X to exclude them.\nInput: "
            )
            .strip()
            .lower()
        )
        EXCLUDE_DUPLICATES_FLAG = exclude_duplicates_choice == "x"
        if EXCLUDE_DUPLICATES_FLAG:
            print(f"{duplicate_count} lines excluded.")

# Apply all filtering
main_df = DataProcessor.apply_filtering(
    main_df,
    exclude_percent=not include_percent_flag,
    exclude_backslash_entries=EXCLUDE_BACKSLASHES_FLAG,
    exclude_duplicate_entries=EXCLUDE_DUPLICATES_FLAG,
)

# Check if we have any data after filtering
if len(main_df) == 0:
    print("No data remaining after filtering. Exiting.")
    exit(1)

# Step 5: Date filtering and period selection
try:
    available_months = (
        main_df.groupby(["Year", "Month"])
        .size()
        .reset_index(name="count")
        .sort_values(["Year", "Month"])
    )
    latest_year = available_months["Year"].max()
    latest_month = available_months[available_months["Year"] == latest_year][
        "Month"
    ].max()
except Exception as e:
    print(f"Error determining available months: {e}")
    exit(1)

# Default to latest month, but override with initial_date_detected if provided
if initial_date_detected:
    try:
        selected_year, selected_month, selected_day = DateProcessor.parse_compact_date(
            initial_date_detected
        )
        if selected_month is None:
            period_start_date = pd.to_datetime(f"{selected_year}-01-01").date()
            period_end_date = pd.to_datetime(f"{selected_year}-12-31").date()
        elif selected_day is None:
            period_start_date = pd.to_datetime(
                f"{selected_year}-{selected_month}-01"
            ).date()
            period_end_date = (
                pd.to_datetime(f"{selected_year}-{selected_month}-01")
                + pd.offsets.MonthEnd(0)
            ).date()
        else:
            target_date = pd.to_datetime(
                f"{selected_year}-{selected_month:02d}-{selected_day:02d}"
            )
            period_start_date = target_date.date()
            period_end_date = target_date.date()
    except ValueError:
        period_start_date = pd.to_datetime(f"{latest_year}-{latest_month}-01").date()
        period_end_date = (
            pd.to_datetime(f"{latest_year}-{latest_month}-01") + pd.offsets.MonthEnd(0)
        ).date()
else:
    period_start_date = pd.to_datetime(f"{latest_year}-{latest_month}-01").date()
    period_end_date = (
        pd.to_datetime(f"{latest_year}-{latest_month}-01") + pd.offsets.MonthEnd(0)
    ).date()

if use_detailed_options_flag:
    print("\nAvailable data by month:")
    for _, row in available_months.iterrows():
        month_abbr = (
            pd.to_datetime(f"{row['Year']}-{row['Month']}-01").strftime("%b").upper()
        )
        MARKER_TEXT = (
            " (LATEST)"
            if (row["Year"] == latest_year and row["Month"] == latest_month)
            else ""
        )
        print(
            f"{row['Year']} {row['Month']:02d} [{month_abbr}]: {row['count']:>3} messages{MARKER_TEXT}"
        )

    print(f"\nDefault: {str(latest_year)} {latest_month:02d} (latest available)")
    print("\nInput NULL to process latest month,")
    print("or else specify a TIME PERIOD (optionally followed by MARKER):")
    print("• Year: '23' (all of 2023)")
    print("• Month: '2307' (July 2023)")
    print("• Month range: '2309 2311' (September through November 2024)")
    print("• Date range: '240115 240424' (Jan 15 through Apr 24, 2024)")
    print("• With marker: '2407 d' or '2407 d!' (July 2024 with diamond markers)")

    # Date input processing
    while True:
        user_date_input = input("\nInput: ").strip()

        # Check for BW mode in the input
        if "bw" in user_date_input.lower():
            BW_MODE_FLAG = True
            user_date_input = user_date_input.lower().replace("bw", "").strip()
            print("Light background mode enabled.")

        try:
            (
                df_filtered,
                period_start_date,
                period_end_date,
                DATE_DESCRIPTION,
                secret_marker_detected,
                include_percent_flag,
            ) = process_date_range_input(
                user_date_input,
                main_df,
                latest_year,
                latest_month,
                include_percent_flag,
                EXCLUDE_BACKSLASHES_FLAG,
                EXCLUDE_DUPLICATES_FLAG,
                df_raw,
            )

            print(f"Processing {len(df_filtered)} messages for {DATE_DESCRIPTION}.")
            break

        except (ValueError, IndexError):
            print("Invalid input. Try again.")
            continue
else:
    # Express mode
    if initial_date_detected:
        try:
            selected_year, selected_month, selected_day = (
                DateProcessor.parse_compact_date(initial_date_detected)
            )
            if selected_month is None:
                df_filtered = main_df[main_df["Year"] == selected_year].copy()
                DATE_DESCRIPTION = str(selected_year)
            elif selected_day is None:
                df_filtered = main_df[
                    (main_df["Year"] == selected_year)
                    & (main_df["Month"] == selected_month)
                ].copy()
                DATE_DESCRIPTION = pd.to_datetime(
                    f"{selected_year}-{selected_month}-01"
                ).strftime("%B %Y")
            else:
                target_date = pd.to_datetime(
                    f"{selected_year}-{selected_month:02d}-{selected_day:02d}"
                )
                df_filtered = main_df[
                    main_df["Datetime"].dt.date == target_date.date()
                ].copy()
                DATE_DESCRIPTION = target_date.strftime("%B %d, %Y")
        except ValueError:
            df_filtered = main_df[
                (main_df["Year"] == latest_year) & (main_df["Month"] == latest_month)
            ].copy()
            DATE_DESCRIPTION = pd.to_datetime(
                f"{latest_year}-{latest_month}-01"
            ).strftime("%B %Y")
    else:
        df_filtered = main_df[
            (main_df["Year"] == latest_year) & (main_df["Month"] == latest_month)
        ].copy()
        DATE_DESCRIPTION = pd.to_datetime(f"{latest_year}-{latest_month}-01").strftime(
            "%B %Y"
        )
    print(f"Processing {len(df_filtered)} messages for {DATE_DESCRIPTION}.")

# Use filtered data for the rest of the processing
main_df = df_filtered

# Check if we have any data for the selected period
if len(main_df) == 0:
    print(f"No data found for the selected period: {DATE_DESCRIPTION}")
    exit(1)

# Step 6: Setup data structures for plotting
complete_date_range = pd.date_range(
    start=period_start_date, end=period_end_date, freq="D"
)
complete_dates = [d.date() for d in complete_date_range]

# Determine display mode
is_short_period = DateProcessor.is_short_period(period_start_date, period_end_date)
is_intermediate_period = DateProcessor.is_intermediate_period(
    period_start_date, period_end_date
)

# Setup date mapping for timeline plot
dates_sorted = sorted(complete_dates)
date_to_y_mapping = {date_val: i for i, date_val in enumerate(dates_sorted[::-1])}
main_df["y"] = main_df["Date"].map(date_to_y_mapping)

# Step 7: Setup prefix grouping and styling
(
    group_to_prefix_mapping,
    prefix_color_mapping,
    prefix_alpha_timeline_mapping,
    prefix_alpha_bar_mapping,
) = StyleProcessor.setup_prefix_grouping(
    main_df, is_short_period, is_intermediate_period
)

# Step 8: Create plots
PERIOD_TYPE = (
    "Short"
    if is_short_period
    else ("Intermediate" if is_intermediate_period else "Long")
)
print(
    f"""Period type: {PERIOD_TYPE} ({(period_end_date - period_start_date).days + 1}
      days)\n\nReports are now being generated."""
)

# Timeline scatter plot
PlotCreator.create_timeline_plot(
    main_df,
    dates_sorted,
    date_to_y_mapping,
    group_to_prefix_mapping,
    prefix_color_mapping,
    prefix_alpha_timeline_mapping,
    is_short_period,
    BW_MODE_FLAG,
    period_start_date,
    period_end_date,
    include_percent_flag,
    secret_marker_detected,
    is_intermediate_period,
)
print("\nReport (type 1) generated successfully.")

# Stacked bar chart
PlotCreator.create_bar_chart(
    main_df,
    complete_dates,
    group_to_prefix_mapping,
    prefix_color_mapping,
    prefix_alpha_bar_mapping,
    is_short_period,
    BW_MODE_FLAG,
    period_start_date,
    period_end_date,
    include_percent_flag,
    is_intermediate_period,
)
print("Report (type 2) generated successfully.")
print("Script has completed.")

# NEET INTEL / JWSCC
# HFGCS EAM PLOT AND CHART GENERATOR
# PUBLIC BRANCH – v0.1 — released 250819
# IDE may throw errors; deal with it
# CONTACT 02_0279@PROTON.ME FOR ENQUIRIES
