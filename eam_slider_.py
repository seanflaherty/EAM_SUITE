"""This module reports on a slider comparison between EAMs."""

import re
import os
import glob
import pandas as pd


def clean_messages(messages):
    """Cleans data to remove duplicates and invalid characters."""
    # Remove duplicates by converting to a set and back to a list
    unique_messages = list(set(messages))

    # Remove messages containing any characters outside of A-Z or 0-9
    valid_messages = [msg for msg in unique_messages if re.match(r"^[A-Z0-9]+$", msg)]

    return valid_messages


def read_csv_data(filename):
    """Read and filter data from CSV file."""
    try:
        df = pd.read_csv(filename)

        # Check if required columns exist
        required_columns = ["Q", "PR", "MESSAGE", "DATE"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing required columns in CSV: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()

        # Filter out messages with '\' or '*' in column Q
        df_filtered = df[
            ~df["Q"].astype(str).str.contains("[\\\\*]", na=False, regex=True)
        ]

        # Also filter out messages with non-alphanumeric characters in column PR
        # Keep only rows where PR contains only letters and numbers
        df_filtered = df_filtered[
            df_filtered["PR"].astype(str).str.match(r"^[A-Za-z0-9]+$", na=False)
        ]

        # Remove rows where MESSAGE, PR, or DATE is NaN
        df_filtered = df_filtered.dropna(subset=["MESSAGE", "PR", "DATE"])

        return df_filtered
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        print("Please ensure SHORTWAVES.csv is in the same directory as this script.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file {filename} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        print("Please check that the CSV file is properly formatted.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()


def get_recent_prefixes(df, top_n=10):
    """Get the most recent prefixes with their date ranges and statistics."""
    if df.empty:
        return []

    # Group by prefix and get statistics
    prefix_stats = []

    for prefix in df["PR"].unique():
        prefix_data = df[df["PR"] == prefix].copy()

        # Convert DATE to datetime for proper sorting
        prefix_data["DATE"] = pd.to_datetime(prefix_data["DATE"])

        # Get date range
        min_date = prefix_data["DATE"].min()
        max_date = prefix_data["DATE"].max()

        # Format dates as YYMMDD
        min_date_str = min_date.strftime("%y%m%d")
        max_date_str = max_date.strftime("%y%m%d")

        # Get message count (unique messages)
        unique_messages = prefix_data["MESSAGE"].nunique()

        # Get message length range (use LN column if available, otherwise calculate)
        if "LN" in prefix_data.columns and not prefix_data["LN"].isna().all():
            min_length = int(prefix_data["LN"].min())
            max_length = int(prefix_data["LN"].max())
        else:
            lengths = prefix_data["MESSAGE"].str.len()
            min_length = lengths.min()
            max_length = lengths.max()

        prefix_stats.append(
            {
                "prefix": prefix,
                "latest_date": max_date,
                "min_date_str": min_date_str,
                "max_date_str": max_date_str,
                "message_count": unique_messages,
                "min_length": min_length,
                "max_length": max_length,
            }
        )

    # Sort by latest date (most recent first)
    prefix_stats.sort(key=lambda x: x["latest_date"], reverse=True)

    return prefix_stats[:top_n]


def display_recent_prefixes(prefix_stats):
    """Display the recent prefixes in the requested format."""
    print("\nTop 10 most-recent prefixes (by latest DATE):")
    for i, stats in enumerate(prefix_stats, 1):
        interval = (
            f"{stats['min_date_str']} — {stats['max_date_str']}"
            if stats["min_date_str"] != stats["max_date_str"]
            else stats["max_date_str"]
        )
        length_range = (
            f"{stats['min_length']}-{stats['max_length']}"
            if stats["min_length"] != stats["max_length"]
            else str(stats["min_length"])
        )

        print(
            f"{i:2d}. {stats['prefix']:<3} (interval: {interval:<15}",
            " messages: {stats['message_count']:>3}, message lengths: {length_range})",
        )


def slide_compare_multiple_grouped(message_data, offset=None, min_matches=6):
    """
    Perform sliding comparison for each group based on the first two characters.
    - If offset is specified as an integer, only compare with that fixed offset.
    - If offset is None, allow sliding across the entire message.
    - Only report results with at least `min_matches`.
    - message_data is a list of tuples: (message, date, time)
    """
    # Extract just the messages for cleaning
    messages = [data[0] for data in message_data]
    cleaned_messages = clean_messages(messages)

    # Create a mapping from cleaned message to original data
    message_to_data = {}
    for msg, date, time in message_data:
        if msg in cleaned_messages:
            message_to_data[msg] = (date, time)

    if len(cleaned_messages) < 2:
        print("Not enough valid messages to compare.")
        return []

    print(f"Analyzing {len(cleaned_messages)} unique valid messages...")

    max_len = max(len(msg) for msg in cleaned_messages)
    best_match_results = []

    # Compare each message with every other message
    for i in range(len(cleaned_messages)):
        base_msg = cleaned_messages[i]
        base_date, base_time = message_to_data[base_msg]
        aligned_result = list(base_msg.ljust(max_len, "-"))

        for j in range(i + 1, len(cleaned_messages)):
            msg = cleaned_messages[j]
            msg_date, msg_time = message_to_data[msg]

            len_msg = len(msg)
            best_offset = 0
            max_matches = 0
            best_match_alignment = []

            # Determine the offsets to use (fixed or sliding)
            if offset is not None:
                offsets = [offset]  # Fixed offset
            else:
                offsets = list(
                    range(-(len_msg - 1), max_len)
                )  # Sliding across all offsets

            for current_offset in offsets:
                matches = []
                match_count = 0

                for k in range(max_len):
                    l = k - current_offset
                    if 0 <= l < len_msg:
                        if aligned_result[k] == msg[l]:
                            # Exact match
                            matches.append(msg[l])
                            match_count += 1
                        else:
                            matches.append("-")
                    else:
                        matches.append("-")

                # Update if this alignment has more matches than the previous best
                if match_count > max_matches:
                    max_matches = match_count
                    best_offset = current_offset
                    best_match_alignment = (
                        matches  # Store best alignment as list for later output
                    )

            # Only store results with matches greater than or equal to the minimum
            if max_matches >= min_matches:
                best_match_results.append(
                    (
                        base_msg[:6],
                        msg[:6],
                        "".join(best_match_alignment),
                        best_offset,
                        max_matches,
                        base_date,
                        base_time,
                        msg_date,
                        msg_time,
                    )
                )

    return best_match_results


def check_existing_files(prefix, current_date):
    """Check for existing output files and handle them based on date comparison."""
    pattern = f"eam_slider_output_{prefix}_*.txt"
    existing_files = glob.glob(pattern)

    if not existing_files:
        return True  # No existing files, proceed

    current_date_str = current_date.strftime("%y%m%d")
    current_date_int = int(current_date_str)

    files_to_delete = []
    newer_files = []

    for file in existing_files:
        # Extract date from filename using regex
        match = re.search(
            rf"eam_slider_output_{re.escape(prefix)}_(\d{{6}})\.txt", file
        )
        if match:
            file_date_str = match.group(1)
            file_date_int = int(file_date_str)

            if file_date_int <= current_date_int:
                files_to_delete.append(file)
            else:
                newer_files.append((file, file_date_str))

    # If there are newer files, warn user
    if newer_files:
        print("\n⚠️  WARNING: Found existing output file(s) with later dates:")
        for file, date_str in newer_files:
            print(f"   {file} (date: {date_str})")

        response = (
            input(
                f"Current analysis is for {current_date_str}. Continue anyway? (y/N): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            print("Analysis cancelled.")
            return False

    # Delete older/same date files
    if files_to_delete:
        print(
            f"\nDeleting {len(files_to_delete)} older output file(s) for prefix '{prefix}':"
        )
        for file in files_to_delete:
            try:
                os.remove(file)
                print(f"   Deleted: {file}")
            except OSError as e:
                print(f"   Error deleting {file}: {e}")

    return True


def write_results_to_file(results, prefix, latest_date):
    """Write results to file with prefix-specific filename including date."""
    date_str = latest_date.strftime("%y%m%d")
    filename = f"eam_slider_output_{prefix}_{date_str}.txt"

    # Sort results by total matches in descending order
    sorted_results = sorted(results, key=lambda x: x[4], reverse=True)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(
            f"Best Match Results for Prefix '{prefix}' (Sorted by Total Matches):\n"
        )
        file.write(
            f"Total comparisons with {len(sorted_results)} matches above threshold:\n\n"
        )

        for result in sorted_results:
            base_serial, compared_serial, best_alignment, best_offset, max_matches = (
                result[:5]
            )
            if len(result) >= 9:  # Check if date/time info is available
                base_date, base_time, msg_date, msg_time = result[5:9]

                # Format dates and times
                base_date_str = pd.to_datetime(base_date).strftime("%y%m%d")
                msg_date_str = pd.to_datetime(msg_date).strftime("%y%m%d")
                base_time_str = str(base_time).zfill(
                    4
                )  # Ensure 4 digits with leading zeros
                msg_time_str = str(msg_time).zfill(4)

                # Format time as HH:MM
                base_time_formatted = f"{base_time_str[:2]}:{base_time_str[2:]}"
                msg_time_formatted = f"{msg_time_str[:2]}:{msg_time_str[2:]}"

                file.write(
                    f"""Base Message: {base_serial} [{base_date_str} {base_time_formatted}],
                     Compared with: {compared_serial} [{msg_date_str} {msg_time_formatted}]\n"""
                )
            else:
                file.write(
                    f"Base Message: {base_serial}, Compared with: {compared_serial}\n"
                )

            file.write(f"Best Offset: {best_offset}, Total Matches: {max_matches}\n")
            file.write(f"Best Alignment: {best_alignment}\n\n")

    return filename


def main():
    """Main execution function."""
    csv_file_path = "SHORTWAVES.csv"

    # Read CSV data
    print("Reading CSV data...")
    df = read_csv_data(csv_file_path)

    if df.empty:
        print("No data available. Exiting.")
        return

    # Get and display recent prefixes
    recent_prefixes = get_recent_prefixes(df)
    display_recent_prefixes(recent_prefixes)

    # Get user selection
    while True:
        user_input = (
            input("\nEnter a prefix to analyze (or press Enter to exit): ")
            .strip()
            .upper()
        )

        if not user_input:
            print("Exiting.")
            break

        # Check if prefix exists in data
        prefix_data = df[df["PR"] == user_input]
        if prefix_data.empty:
            print(f"No messages found for prefix '{user_input}'. Try again.")
            continue

        # Get unique messages for this prefix with their date/time data
        message_data = []
        for _, row in prefix_data.iterrows():
            message = row["MESSAGE"]
            date = row["DATE"]
            time = row.get(
                "UTC", "0000"
            )  # Default to '0000' if UTC column doesn't exist
            message_data.append((message, date, time))

        # Remove duplicates based on message content while preserving date/time info
        unique_message_data = []
        seen_messages = set()
        for msg, date, time in message_data:
            if msg not in seen_messages:
                unique_message_data.append((msg, date, time))
                seen_messages.add(msg)

        latest_date = pd.to_datetime(prefix_data["DATE"]).max()
        print(
            f"\nFound {len(unique_message_data)} unique messages for prefix '{user_input}'"
        )

        # Get analysis parameters
        offset_input = input(
            "Enter a specific offset (number), or press Enter for sliding offset: "
        ).strip()
        try:
            offset_value = int(offset_input) if offset_input else None
        except ValueError:
            print(f"Invalid input '{offset_input}'. Using sliding offset.")
            offset_value = None

        min_matches_input = input(
            "Enter minimum number of matches to report, or press Enter for default (6): "
        ).strip()
        try:
            min_matches_value = int(min_matches_input) if min_matches_input else 6
        except ValueError:
            print(f"Invalid input '{min_matches_input}'. Using default of 6 matches.")
            min_matches_value = 6

        # Perform analysis
        print("\nPerforming sliding comparison analysis...")
        best_match_results = slide_compare_multiple_grouped(
            unique_message_data, offset=offset_value, min_matches=min_matches_value
        )

        if best_match_results:
            # Check for existing files and handle them
            if not check_existing_files(user_input, latest_date):
                continue  # User cancelled, go back to prefix selection

            # Write results to file
            output_filename = write_results_to_file(
                best_match_results, user_input, latest_date
            )
            print(
                f"""Results with at least {min_matches_value}
                matches have been written to {output_filename}"""
            )
            print(
                f"Found {len(best_match_results)} message pairs with sufficient matches."
            )

            # Display results to console
            print(f"\n--- Results for Prefix '{user_input}' ---")
            sorted_results = sorted(
                best_match_results, key=lambda x: x[4], reverse=True
            )

            for result in sorted_results:
                (
                    base_serial,
                    compared_serial,
                    best_alignment,
                    best_offset,
                    max_matches,
                ) = result[:5]
                if len(result) >= 9:  # Check if date/time info is available
                    base_date, base_time, msg_date, msg_time = result[5:9]

                    # Format dates and times
                    base_date_str = pd.to_datetime(base_date).strftime("%y%m%d")
                    msg_date_str = pd.to_datetime(msg_date).strftime("%y%m%d")
                    base_time_str = str(base_time).zfill(
                        4
                    )  # Ensure 4 digits with leading zeros
                    msg_time_str = str(msg_time).zfill(4)

                    # Format time as HH:MM
                    base_time_formatted = f"{base_time_str[:2]}:{base_time_str[2:]}"
                    msg_time_formatted = f"{msg_time_str[:2]}:{msg_time_str[2:]}"

                    print(
                        f"""\nBase: {base_serial} [{base_date_str} {base_time_formatted}]
                        , Compared: {compared_serial} [{msg_date_str} {msg_time_formatted}]"""
                    )
                else:
                    print(f"\nBase: {base_serial}, Compared: {compared_serial}")

                print(f"Offset: {best_offset}, Matches: {max_matches}")
                print(f"Alignment: {best_alignment}")
        else:
            print(f"No message pairs found with at least {min_matches_value} matches.")

        print("-" * 50)


if __name__ == "__main__":
    main()

# NEET INTEL / JWSCC
# PUBLIC BRANCH – v0.1 — released 250819
# IDE may throw errors; deal with it
# CONTACT 02_0279@PROTON.ME FOR ENQUIRIES
