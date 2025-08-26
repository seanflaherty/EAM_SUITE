"""Character Frequency Analysis module."""

from typing import List, Tuple, Dict, Optional
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class CharacterAnalyzer:
    """
    A comprehensive character frequency analyzer for text data.

    :param csv_file_path
    :type csv_file_path: str
    :param df
    :type df: Optional[pd.DataFrame] or none
    :param processed_messages
    :type processed_message: List[str]

    """

    def __init__(self, file_path: str):
        """Constructor method."""
        self.csv_file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.processed_messages: List[str] = []

        # Set dark theme for matplotlib
        plt.style.use("dark_background")

    def load_data(self) -> bool:
        """Load and preprocess data from CSV file."""
        try:
            # Read CSV file with various encoding options to handle potential issues
            encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

            for encoding in encodings_to_try:
                try:
                    self.df = pd.read_csv(self.csv_file_path, encoding=encoding)
                    print(
                        f"""Successfully loaded {len(self.df)} rows from
                          {self.csv_file_path} using {encoding} encoding"""
                    )
                    return True
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, try with error handling
            self.df = pd.read_csv(self.csv_file_path, encoding="utf-8", errors="ignore")
            print(
                f"""Successfully loaded {len(self.df)} rows from
                 {self.csv_file_path} (with error handling)"""
            )
            return True

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

    def filter_data(self) -> None:
        """Apply all data filtering steps."""
        if "MESSAGE" not in self.df.columns:
            raise ValueError("'MESSAGE' column not found in the CSV file")

        initial_count = len(self.df)

        # Filter messages with forbidden digits
        self._filter_forbidden_digits()

        # Filter problematic characters in specific columns
        self._filter_problematic_chars()

        # Remove duplicates
        self._remove_duplicates()

        print(f"Final dataset: {len(self.df)} unique messages")

    def _filter_forbidden_digits(self) -> None:
        """Filter out messages containing digits 0, 1, 8, or 9."""
        forbidden_messages = self.df[
            self.df["MESSAGE"].astype(str).str.contains("[0189]", na=False)
        ]

        if not forbidden_messages.empty and "DATE" in self.df.columns:
            forbidden_dates = forbidden_messages["DATE"].dropna().unique()
            if len(forbidden_dates) > 0:
                print(
                    f"""Found messages with forbidden digits (0,1,8,9) on dates:
                     {', '.join(map(str, forbidden_dates))}"""
                )

        initial_count = len(self.df)
        self.df = self.df[
            ~self.df["MESSAGE"].astype(str).str.contains("[0189]", na=False)
        ]
        filtered_count = initial_count - len(self.df)
        print(f"Filtered out {filtered_count} messages containing digits 0, 1, 8, or 9")

    def _filter_problematic_chars(self) -> None:
        """Filter out rows with problematic characters in specific columns."""
        filters = [
            ("Q", "\\\\", "backslashes"),
            ("SOURCE", "`", "backticks"),
            ("MESSAGE", "[_.?]", "underscores, periods, or question marks"),
        ]

        for column, pattern, description in filters:
            if column in self.df.columns:
                initial_count = len(self.df)
                self.df = self.df[
                    ~self.df[column].astype(str).str.contains(pattern, na=False)
                ]
                filtered_count = initial_count - len(self.df)
                print(
                    f"Filtered out {filtered_count} rows with {description} in {column} column"
                )

    def _remove_duplicates(self) -> None:
        """Remove duplicate messages."""
        if self.df is not None:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=["MESSAGE"])
            duplicate_count = initial_count - len(self.df)
            print(f"Removed {duplicate_count} duplicate messages")
        else:
            print("Warning: DataFrame is None, cannot remove duplicates.")

    def preprocess_messages(
        self,
        include_first_two: bool = False,
        exclude_consecutive: bool = False,
        consecutive_count: int = 4,
    ) -> List[str]:
        """Preprocess messages based on specified criteria."""
        processed_messages = []
        excluded_count = 0

        for _, row in self.df.iterrows():
            message = row["MESSAGE"]
            if pd.isna(message) or not isinstance(message, str):
                continue

            # Apply consecutive character filter if requested
            if exclude_consecutive and self._has_n_consecutive_chars(
                message, consecutive_count
            ):
                excluded_count += 1
                continue

            # Apply first two characters rule
            processed_message = (
                message
                if include_first_two
                else (message[2:] if len(message) > 2 else "")
            )
            processed_messages.append(processed_message)

        # Report filtering results
        if include_first_two:
            print("Including first 2 characters in analysis")
        else:
            print("Ignoring first 2 characters in analysis")

        if exclude_consecutive:
            print(
                f"""Excluded {excluded_count} messages with
                 {consecutive_count}+ consecutive identical characters"""
            )
            print(f"Analyzing {len(processed_messages)} messages after filtering")

        self.processed_messages = processed_messages
        return processed_messages

    @staticmethod
    def _has_n_consecutive_chars(text: str, n: int) -> bool:
        """Check if text has n consecutive identical characters."""
        if len(text) < n:
            return False

        for i in range(len(text) - n + 1):
            if all(text[i] == text[i + j] for j in range(n)):
                return True
        return False

    @staticmethod
    def _find_consecutive_chars(text: str, min_count: int = 4) -> List[Tuple[str, int]]:
        """Find all characters that appear min_count+ times consecutively."""
        if len(text) < min_count:
            return []

        consecutive_chars = []
        i = 0
        while i < len(text) - min_count + 1:
            current_char = text[i]
            count = 1
            j = i + 1

            # Count consecutive occurrences
            while j < len(text) and text[j] == current_char:
                count += 1
                j += 1

            # If min_count or more consecutive, record it
            if count >= min_count:
                consecutive_chars.append((current_char, count))
                i = j  # Skip past this sequence
            else:
                i += 1

        return consecutive_chars

    def _create_statistical_chart(
        self,
        chart_data: List[Tuple[str, int]],
        chart_title: str,
        bar_color: str = "#4A9EFF",
        max_displayable_items: int = 50,
    ) -> None:
        """Create a standardized statistical chart with outlier detection and dark theme."""
        if len(chart_data) > max_displayable_items:
            print(
                f"""\n(Too many items to display chart - showing all
                  {len(chart_data)} in text format)"""
            )
            return

        chart_items, frequency_counts = zip(*chart_data)

        # Determine chart size
        if len(chart_data) <= 20:
            chart_figure_width, label_fontsize, x_label_rotation = 14, 10, 0
        elif len(chart_data) <= 35:
            chart_figure_width, label_fontsize, x_label_rotation = 18, 8, 45
        else:
            chart_figure_width, label_fontsize, x_label_rotation = 24, 7, 45

        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(chart_figure_width, 8))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#2d2d2d")

        chart_bars = ax.bar(
            range(len(chart_items)),
            frequency_counts,
            color=bar_color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add statistical reference lines with bright colors for dark theme
        counts_mean_freq = np.mean(frequency_counts)
        counts_std_freq = np.std(frequency_counts)

        ax.axhline(
            y=counts_mean_freq,
            color="#FF6B6B",
            linestyle="--",
            alpha=0.9,
            linewidth=2,
            label=f"Mean: {counts_mean_freq:.1f}",
        )
        ax.axhline(
            y=counts_mean_freq + counts_std_freq,
            color="#FFD93D",
            linestyle=":",
            alpha=0.9,
            linewidth=2,
            label=f"Mean + 1σ: {counts_mean_freq + counts_std_freq:.1f}",
        )
        ax.axhline(
            y=counts_mean_freq - counts_std_freq,
            color="#FFD93D",
            linestyle=":",
            alpha=0.9,
            linewidth=2,
            label=f"Mean - 1σ: {counts_mean_freq - counts_std_freq:.1f}",
        )

        # Highlight outliers
        self._highlight_outliers(chart_bars, frequency_counts, chart_data)

        # Customize chart with light colors for dark theme
        ax.set_title(chart_title, fontsize=16, fontweight="bold", color="white", pad=20)
        ax.set_xlabel("Items", fontsize=12, color="white")
        ax.set_ylabel("Frequency", fontsize=12, color="white")

        # Style the legend for dark theme
        legend = ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            ncol=3,
            facecolor="#3d3d3d",
            edgecolor="white",
            framealpha=0.9,
        )
        legend.get_frame().set_linewidth(1)
        for text in legend.get_texts():
            text.set_color("white")

        # Set labels with capitalization for alphabetic/alphanumeric characters
        chart_display_items = []
        for chart_item in chart_items:
            if chart_item == " ":
                chart_display_items.append("SPACE")
            elif chart_item.isalnum():  # Capitalize alphanumeric characters
                chart_display_items.append(chart_item.upper())
            else:
                chart_display_items.append(chart_item)

        ax.set_xticks(range(len(chart_items)))
        ax.set_xticklabels(
            [f"'{chart_item}'" for chart_item in chart_display_items],
            rotation=x_label_rotation,
            color="white",
        )

        # Style tick parameters for dark theme
        ax.tick_params(axis="both", colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["left"].set_color("white")

        # Add value labels on bars (only if not too many)
        if len(chart_data) <= 30:
            for chart_bar, frequency_count in zip(chart_bars, frequency_counts):
                ax.text(
                    chart_bar.get_x() + chart_bar.get_width() / 2,
                    chart_bar.get_height() + max(frequency_counts) * 0.01,
                    str(frequency_count),
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    color="white",
                    fontweight="bold",
                )

        ax.grid(axis="y", alpha=0.3, color="gray")
        plt.tight_layout()
        plt.show()

    def _highlight_outliers(
        self, chart_bars, frequency_counts: List[int], chart_data: List[Tuple[str, int]]
    ) -> None:
        """Highlight outliers and color bars by type with dark theme colors."""
        counts_q1, counts_q3 = np.percentile(frequency_counts, [25, 75])
        counts_iqr = counts_q3 - counts_q1
        outlier_threshold_high = counts_q3 + 1.5 * counts_iqr
        outlier_threshold_low = counts_q1 - 1.5 * counts_iqr

        for bar_index, (data_item, item_count) in enumerate(chart_data):
            if (
                item_count > outlier_threshold_high
                or item_count < outlier_threshold_low
            ):
                chart_bars[bar_index].set_color("#FF4757")  # Bright red for outliers
                chart_bars[bar_index].set_alpha(0.9)
            elif data_item.isdigit():
                chart_bars[bar_index].set_color("#4A9EFF")  # Bright blue for digits
            elif data_item.isalpha():
                chart_bars[bar_index].set_color("#2ED573")  # Bright green for letters
            else:
                chart_bars[bar_index].set_color("#FFA726")  # Bright orange for symbols

    def _print_statistics(
        self, stats_data: List[Tuple[str, int]], stats_title: str
    ) -> None:
        """Print comprehensive statistics for the data."""
        if not stats_data:
            return

        stats_items, stats_counts = zip(*stats_data)

        print(f"\nStatistical Summary for {stats_title}:")
        print(f"Unique items: {len(stats_data)}")
        print(f"Mean frequency: {np.mean(stats_counts):.2f}")
        print(f"Median frequency: {np.median(stats_counts):.2f}")
        print(f"Standard deviation: {np.std(stats_counts):.2f}")
        print(f"Range: {max(stats_counts) - min(stats_counts)}")

        # Quartile analysis
        stats_q1, stats_q3 = np.percentile(stats_counts, [25, 75])
        stats_iqr = stats_q3 - stats_q1
        print(f"1st Quartile (Q1): {stats_q1:.2f}")
        print(f"3rd Quartile (Q3): {stats_q3:.2f}")
        print(f"Interquartile Range (IQR): {stats_iqr:.2f}")

        # Outliers
        stats_outlier_high = stats_q3 + 1.5 * stats_iqr
        stats_outlier_low = stats_q1 - 1.5 * stats_iqr
        stats_outliers = [
            (stats_item, stats_count)
            for stats_item, stats_count in stats_data
            if stats_count > stats_outlier_high or stats_count < stats_outlier_low
        ]

        if stats_outliers:
            print("\nOutliers (unusual frequencies):")
            for stats_item, stats_count in stats_outliers[:10]:
                stats_display_item = "SPACE" if stats_item == " " else stats_item
                print(f"'{stats_display_item}': {stats_count} times")

    def analyze_character_context(
        self, target_character: str, character_position: str
    ) -> None:
        """Analyze characters that appear before or after a target character."""
        if character_position not in ["before", "after"]:
            raise ValueError("Position must be 'before' or 'after'")

        context_char_counts = Counter()
        total_context_occurrences = 0

        for processed_message in self.processed_messages:
            if (
                pd.isna(processed_message)
                or not isinstance(processed_message, str)
                or len(processed_message) < 2
            ):
                continue

            if character_position == "after":
                for char_index in range(len(processed_message) - 1):
                    if (
                        processed_message[char_index].lower()
                        == target_character.lower()
                    ):
                        following_char = processed_message[char_index + 1]
                        context_char_counts[following_char] += 1
                        total_context_occurrences += 1
            else:  # before
                for char_index in range(1, len(processed_message)):
                    if (
                        processed_message[char_index].lower()
                        == target_character.lower()
                    ):
                        preceding_char = processed_message[char_index - 1]
                        context_char_counts[preceding_char] += 1
                        total_context_occurrences += 1

        if not context_char_counts:
            print(f"No characters found {character_position} '{target_character}'!")
            return

        # Sort by frequency
        sorted_context_chars = sorted(
            context_char_counts.items(), key=lambda x: x[1], reverse=True
        )

        print(
            f"\nCharacters that appear {character_position.upper()} '{target_character.upper()}':"
        )
        print(f"Total occurrences: {total_context_occurrences}")

        # Show top results
        print(
            f"""\nTop characters {character_position}
              '{target_character.upper()}' (ranked by frequency):"""
        )
        for context_rank, (context_char, context_count) in enumerate(
            sorted_context_chars[:20], 1
        ):
            context_percentage = (context_count / total_context_occurrences) * 100
            context_char_type = self._get_char_type(context_char)
            context_display_char = "SPACE" if context_char == " " else context_char
            print(
                f"""{context_rank:2d}. '{context_display_char}'
                  ({context_char_type}): {context_count} times ({context_percentage:.1f}%)"""
            )

        # Create visualization and statistics
        context_chart_title = f"""Characters Appearing {character_position.title()}
        '{target_character.upper()}' with Statistical Analysis"""
        self._create_statistical_chart(sorted_context_chars, context_chart_title)
        self._print_statistics(
            sorted_context_chars,
            f"characters {character_position} '{target_character.upper()}'",
        )

    @staticmethod
    def _get_char_type(char: str) -> str:
        """Determine the type of character."""
        if char.isdigit():
            return "digit"
        elif char.isalpha():
            return "letter"
        elif char == " ":
            return "space"
        else:
            return "symbol"

    def analyze_consecutive_characters(self, min_consecutive: int = 4) -> None:
        """Analyze which characters appear min_consecutive+ times consecutively."""
        consecutive_char_counts = Counter()
        consecutive_char_max_lengths = {}
        total_consecutive_sequences = 0
        messages_with_consecutive_chars = 0

        for analyzed_message in self.processed_messages:
            if pd.isna(analyzed_message) or not isinstance(analyzed_message, str):
                continue

            consecutive_sequences_found = self._find_consecutive_chars(
                analyzed_message, min_consecutive
            )
            if consecutive_sequences_found:
                messages_with_consecutive_chars += 1

            for consecutive_char, consecutive_length in consecutive_sequences_found:
                consecutive_char_counts[consecutive_char] += 1
                total_consecutive_sequences += 1
                # Track the maximum length seen for each character
                if (
                    consecutive_char not in consecutive_char_max_lengths
                    or consecutive_length
                    > consecutive_char_max_lengths[consecutive_char]
                ):
                    consecutive_char_max_lengths[consecutive_char] = consecutive_length

        if not consecutive_char_counts:
            print(
                f"No characters found with {min_consecutive}+ consecutive occurrences!"
            )
            return

        print("\nConsecutive Character Analysis:")
        print(
            f"""Messages with {min_consecutive}+ consecutive chars:
             {messages_with_consecutive_chars}/{len(self.processed_messages)}"""
        )
        print(f"Total consecutive sequences found: {total_consecutive_sequences}")

        # Sort by frequency
        sorted_consecutive_chars = sorted(
            consecutive_char_counts.items(), key=lambda x: x[1], reverse=True
        )

        print(
            f"""\nCharacters that appear {min_consecutive}+
             times consecutively (ranked by frequency):"""
        )
        for consecutive_rank, (consecutive_char, consecutive_count) in enumerate(
            sorted_consecutive_chars, 1
        ):
            consecutive_max_length = consecutive_char_max_lengths[consecutive_char]
            consecutive_percentage = (
                consecutive_count / total_consecutive_sequences
            ) * 100
            consecutive_char_type = self._get_char_type(consecutive_char)
            print(
                f"""{consecutive_rank:2d}. '{consecutive_char}'
                  ({consecutive_char_type}): {consecutive_count}
                sequences ({consecutive_percentage:.1f}%) - max length: {consecutive_max_length}"""
            )

        # Analyze by type
        self._analyze_consecutive_by_type(
            sorted_consecutive_chars, consecutive_char_max_lengths, min_consecutive
        )

        # Create visualization and statistics
        consecutive_chart_title = f"""Frequency of Characters Appearing {min_consecutive}+
         Times Consecutively with Statistical Analysis"""
        self._create_statistical_chart(
            sorted_consecutive_chars, consecutive_chart_title
        )
        self._print_statistics(
            sorted_consecutive_chars, f"consecutive characters ({min_consecutive}+)"
        )

    def _analyze_consecutive_by_type(
        self,
        sorted_consecutive_chars: List[Tuple[str, int]],
        consecutive_char_max_lengths: Dict[str, int],
        min_consecutive: int,
    ) -> None:
        """Analyze consecutive characters by type (digits, letters, symbols)."""
        consecutive_types = {
            "DIGITS": [
                (consecutive_char, consecutive_count)
                for consecutive_char, consecutive_count in sorted_consecutive_chars
                if consecutive_char.isdigit()
            ],
            "LETTERS": [
                (consecutive_char, consecutive_count)
                for consecutive_char, consecutive_count in sorted_consecutive_chars
                if consecutive_char.isalpha()
            ],
            "SYMBOLS": [
                (consecutive_char, consecutive_count)
                for consecutive_char, consecutive_count in sorted_consecutive_chars
                if not consecutive_char.isalnum()
            ],
        }

        for type_category_name, type_category_chars in consecutive_types.items():
            if type_category_chars:
                print(f"\nTop consecutive {type_category_name} ({min_consecutive}+):")
                total_for_type_category = sum(
                    type_count for _, type_count in type_category_chars
                )
                for type_rank, (type_char, type_count) in enumerate(
                    type_category_chars[:5], 1
                ):
                    type_max_length = consecutive_char_max_lengths[type_char]
                    type_percentage = (type_count / total_for_type_category) * 100
                    print(
                        f"""  {type_rank}. '{type_char}': {type_count} sequences
                        ( {type_percentage:.1f}%) - max length: {type_max_length}"""
                    )

    def analyze_numbers(self, requested_digit_length: int) -> None:
        """Analyze number frequencies for a specific digit length."""
        # Extract all numbers from processed messages
        all_extracted_numbers = []
        for processed_message in self.processed_messages:
            extracted_numbers = (
                re.findall(r"\d+", processed_message)
                if isinstance(processed_message, str)
                else []
            )
            all_extracted_numbers.extend(extracted_numbers)

        if not all_extracted_numbers:
            print("No digit sequences found in the data")
            return

        number_frequency_counts = Counter(all_extracted_numbers)

        # Filter for specific digit length
        filtered_number_counts = {
            number_string: number_count
            for number_string, number_count in number_frequency_counts.items()
            if len(number_string) == requested_digit_length and number_count > 0
        }

        if not filtered_number_counts:
            print(f"\nNo {requested_digit_length}-digit numbers found in the data!")
            return

        range_start_num = (
            0 if requested_digit_length == 1 else 10 ** (requested_digit_length - 1)
        )
        range_end_num = 10**requested_digit_length
        total_possible_numbers = range_end_num - range_start_num
        found_numbers_count = len(filtered_number_counts)

        print(
            f"""\nAnalyzing {requested_digit_length}-digit numbers
              ({range_start_num}-{range_end_num-1}):"""
        )
        print(f"Numbers that appear: {found_numbers_count}/{total_possible_numbers}")
        print(
            f"""Numbers that don't appear:
              {total_possible_numbers - found_numbers_count}/{total_possible_numbers}"""
        )

        # Sort by number value for proper ordering
        sorted_numbers_by_value = sorted(
            filtered_number_counts.items(), key=lambda x: int(x[0])
        )

        # Show top 10 most frequent numbers
        top_frequent_numbers = sorted(
            filtered_number_counts.items(), key=lambda x: x[1], reverse=True
        )
        print(
            f"""\nTop {min(10, len(top_frequent_numbers))}
              most frequent {requested_digit_length}-digit numbers:"""
        )
        for number_string, number_count in top_frequent_numbers[:10]:
            print(f"{number_string}: {number_count} occurrences")

        # Create visualization and statistics
        numbers_chart_title = f""""Frequency of
          {requested_digit_length}-Digit Numbers with Statistical Analysis"""
        self._create_statistical_chart(sorted_numbers_by_value, numbers_chart_title)
        self._print_statistics(
            sorted_numbers_by_value, f"{requested_digit_length}-digit numbers"
        )

    def analyze_letters(self) -> None:
        """Analyze letter frequencies."""
        # Extract all letters from processed messages
        all_extracted_letters = []
        for processed_message in self.processed_messages:
            if isinstance(processed_message, str):
                extracted_letters = re.findall(r"[a-zA-Z]", processed_message.lower())
                all_extracted_letters.extend(extracted_letters)

        if not all_extracted_letters:
            print("No letters found in the data!")
            return

        letter_frequency_counts = Counter(all_extracted_letters)

        # Create frequency data for all letters a-z
        alphabet_letters_az = [
            chr(letter_code) for letter_code in range(ord("a"), ord("z") + 1)
        ]
        found_alphabet_letters = [
            (alphabet_letter, letter_frequency_counts.get(alphabet_letter, 0))
            for alphabet_letter in alphabet_letters_az
            if letter_frequency_counts.get(alphabet_letter, 0) > 0
        ]

        total_letter_occurrences = sum(
            letter_count for _, letter_count in found_alphabet_letters
        )

        print("\nAnalyzing letter frequencies:")
        print(f"Letters that appear: {len(found_alphabet_letters)}/26")
        print(f"Letters that don't appear: {26 - len(found_alphabet_letters)}/26")
        print(f"Total letter occurrences: {total_letter_occurrences}")

        print("\nLetter frequencies:")
        for alphabet_letter, letter_count in found_alphabet_letters:
            letter_percentage = (letter_count / total_letter_occurrences) * 100
            print(
                f"{alphabet_letter.upper()}: {letter_count} occurrences ({letter_percentage:.1f}%)"
            )

        # Show top and bottom frequencies
        sorted_alphabet_letters = sorted(
            found_alphabet_letters, key=lambda x: x[1], reverse=True
        )

        print("\nTop 5 most frequent letters:")
        for alphabet_letter, letter_count in sorted_alphabet_letters[:5]:
            letter_percentage = (letter_count / total_letter_occurrences) * 100
            print(
                f"{alphabet_letter.upper()}: {letter_count} ({letter_percentage:.1f}%)"
            )

        print("\nTop 5 least frequent letters (that appear):")
        for alphabet_letter, letter_count in sorted_alphabet_letters[-5:]:
            letter_percentage = (letter_count / total_letter_occurrences) * 100
            print(
                f"{alphabet_letter.upper()}: {letter_count} ({letter_percentage:.1f}%)"
            )

        # Create visualization and statistics
        letters_chart_title = "Letter Frequency Analysis with Statistical Analysis"
        self._create_statistical_chart(
            found_alphabet_letters, letters_chart_title, bar_color="#2ED573"
        )
        self._print_statistics(found_alphabet_letters, "letters")

    def analyze_two_char_combinations(self) -> None:
        """Analyze all possible 2-character alphanumeric combinations."""
        # Generate all possible 2-character combinations
        # (excluding 0,1,8,9 which are already filtered)
        allowed_chars = list("abcdefghijklmnopqrstuvwxyz") + [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
        ]
        all_possible_combinations = []

        # Generate all combinations
        for first_char in allowed_chars:
            for second_char in allowed_chars:
                all_possible_combinations.append(first_char + second_char)

        print(
            f"Total possible 2-character combinations: {len(all_possible_combinations)}"
        )

        # Extract all 2-character alphanumeric sequences from processed messages
        found_combinations = Counter()
        total_sequences = 0

        for processed_message in self.processed_messages:
            if not isinstance(processed_message, str) or len(processed_message) < 2:
                continue

            # Find all 2-character alphanumeric sequences
            for i in range(len(processed_message) - 1):
                two_char_seq = processed_message[i : i + 2].lower()
                # Check if both characters are alphanumeric and allowed
                if (
                    len(two_char_seq) == 2
                    and all(c.isalnum() for c in two_char_seq)
                    and all(c in allowed_chars for c in two_char_seq)
                ):
                    found_combinations[two_char_seq] += 1
                    total_sequences += 1

        if not found_combinations:
            print("No 2-character alphanumeric combinations found in the data!")
            return

        # Create complete dataset including combinations with 0 occurrences
        complete_combinations = [
            (combo, found_combinations.get(combo, 0))
            for combo in all_possible_combinations
        ]
        found_combinations_only = [
            (combo, count) for combo, count in complete_combinations if count > 0
        ]
        missing_combinations = [
            (combo, count) for combo, count in complete_combinations if count == 0
        ]

        print("\nAnalyzing 2-character alphanumeric combinations:")
        print(
            f"""Combinations that appear:
              {len(found_combinations_only)}/{len(all_possible_combinations)}"""
        )
        print(
            f"""Combinations that don't appear:
              {len(missing_combinations)}/{len(all_possible_combinations)}"""
        )
        print(f"Total occurrences: {total_sequences}")

        # Sort by frequency for analysis
        sorted_combinations = sorted(
            found_combinations_only, key=lambda x: x[1], reverse=True
        )

        # Show top combinations
        print("\nTop 20 most frequent 2-character combinations:")
        for rank, (combo, count) in enumerate(sorted_combinations[:20], 1):
            percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0
            print(
                f"{rank:2d}. '{combo.upper()}': {count} occurrences ({percentage:.2f}%)"
            )

        # Show bottom combinations (that appear)
        if len(sorted_combinations) > 20:
            print("\nBottom 10 least frequent combinations (that appear):")
            for rank, (combo, count) in enumerate(
                sorted_combinations[-10:], len(sorted_combinations) - 9
            ):
                percentage = (
                    (count / total_sequences) * 100 if total_sequences > 0 else 0
                )
                print(
                    f"{rank:2d}. '{combo.upper()}': {count} occurrences ({percentage:.2f}%)"
                )

        # Analyze by type
        self._analyze_combinations_by_type(sorted_combinations, total_sequences)

        # Show some missing combinations
        if missing_combinations:
            print("\nSample of missing combinations (first 20):")
            for combo, _ in missing_combinations[:20]:
                print(f"'{combo.upper()}'", end=" ")
            print()

        # Create visualization and statistics (limit to found combinations for readability)
        combinations_chart_title = (
            "2-Character Alphanumeric Combination Frequencies with Statistical Analysis"
        )
        if len(found_combinations_only) <= 100:  # Only chart if reasonable number
            self._create_statistical_chart(
                sorted_combinations, combinations_chart_title, bar_color="#9B59B6"
            )
        else:
            print(
                f"""\n(Too many combinations ({len(found_combinations_only)})
                  to display chart - showing statistics only)"""
            )

        self._print_statistics(found_combinations_only, "2-character combinations")

    def _analyze_combinations_by_type(
        self, sorted_combinations: List[Tuple[str, int]], total_sequences: int
    ) -> None:
        """Analyze 2-character combinations by type categories."""
        combination_types = {
            "LETTER-LETTER": [
                (combo, count)
                for combo, count in sorted_combinations
                if combo[0].isalpha() and combo[1].isalpha()
            ],
            "DIGIT-DIGIT": [
                (combo, count)
                for combo, count in sorted_combinations
                if combo[0].isdigit() and combo[1].isdigit()
            ],
            "LETTER-DIGIT": [
                (combo, count)
                for combo, count in sorted_combinations
                if combo[0].isalpha() and combo[1].isdigit()
            ],
            "DIGIT-LETTER": [
                (combo, count)
                for combo, count in sorted_combinations
                if combo[0].isdigit() and combo[1].isalpha()
            ],
        }

        print("\nAnalysis by combination type:")
        for type_name, type_combos in combination_types.items():
            if type_combos:
                type_total = sum(count for _, count in type_combos)
                type_percentage = (
                    (type_total / total_sequences) * 100 if total_sequences > 0 else 0
                )
                print(
                    f"""\n{type_name}: {len(type_combos)} different combinations,
                     {type_total} total occurrences ({type_percentage:.2f}%)"""
                )

                # Show top 5 for each type
                for rank, (combo, count) in enumerate(type_combos[:5], 1):
                    combo_percentage = (
                        (count / type_total) * 100 if type_total > 0 else 0
                    )
                    print(
                        f"""  {rank}. '{combo.upper()}':
                         {count} ({combo_percentage:.1f}% of {type_name.lower()})"""
                    )

    def run_interactive_analysis(self) -> None:
        """Run the interactive analysis interface."""
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return

        try:
            self.filter_data()
        except Exception as e:
            print(f"Error filtering data: {e}")
            return

        print("\nWhat would you like to analyze?")
        print(
            "Enter a NUMBER (1, 2, 3, etc.) to analyze digit sequences of that length"
        )
        print("Enter any LETTER to analyze alphabet frequencies (a-z)")
        print(
            """Enter any 2-CHARACTER combination (AB, 23, X7, etc.)
              to analyze all 2-char combinations"""
        )
        print(
            """Enter ASTERISKS (**,***,****,etc.) to analyze characters
              appearing that many times consecutively"""
        )
        print(
            "Enter 'X_' (where X is any letter/number) to see what characters appear AFTER X"
        )
        print(
            "Enter '_X' (where X is any letter/number) to see what characters appear BEFORE X"
        )
        print("Add '+' to INCLUDE the first 2 characters (e.g., '2+' or 'A+')")
        print(
            "Add '-' to EXCLUDE strings with consecutive identical characters (e.g., '2-')"
        )
        print("You can combine modifiers (e.g., '2+-')")
        print(
            """Without modifiers, the first 2 characters 
            are ignored and consecutive chars are included"""
        )
        print("Enter 'quit' or 'exit' to quit the program")

        max_attempts = 10  # Prevent infinite loop
        attempts = 0

        while attempts < max_attempts:
            try:
                user_analysis_input = input("\nEnter your choice: ").strip()

                # Check for exit commands
                if user_analysis_input.lower() in ["quit", "exit", "q"]:
                    print("Exiting analysis. Goodbye!")
                    return

                if not user_analysis_input:
                    print("Please enter a valid choice.")
                    attempts += 1
                    continue

                # Parse modifiers
                include_first_two_chars = "+" in user_analysis_input
                exclude_consecutive_chars = "-" in user_analysis_input
                core_analysis_input = user_analysis_input.replace("+", "").replace(
                    "-", ""
                )

                # Check for asterisk pattern first
                if (
                    core_analysis_input
                    and all(c == "*" for c in core_analysis_input)
                    and len(core_analysis_input) >= 2
                ):
                    consecutive_count = len(core_analysis_input)
                    # For exclusion filter, we need to pass the consecutive count
                    self.preprocess_messages(
                        include_first_two_chars,
                        exclude_consecutive_chars,
                        consecutive_count,
                    )
                    self.analyze_consecutive_characters(consecutive_count)
                    break

                # Preprocess messages based on modifiers (use default 4 for non-asterisk analyses)
                self.preprocess_messages(
                    include_first_two_chars, exclude_consecutive_chars
                )

                # Handle different analysis types
                if self._handle_pattern_analysis(core_analysis_input):
                    break
                elif len(core_analysis_input) == 2 and all(
                    c.isalnum() for c in core_analysis_input
                ):
                    # Any 2-character alphanumeric input triggers combination analysis
                    self.analyze_two_char_combinations()
                    break
                elif core_analysis_input.isdigit():
                    requested_digit_length = int(core_analysis_input)
                    if requested_digit_length >= 1:
                        self.analyze_numbers(requested_digit_length)
                        break
                    else:
                        print("Please enter a number 1 or higher")
                        attempts += 1
                elif core_analysis_input.isalpha() and len(core_analysis_input) == 1:
                    self.analyze_letters()
                    break
                else:
                    self._print_usage_help()
                    attempts += 1

            except KeyboardInterrupt:
                print("\nExiting...")
                return
            except EOFError:
                print("\nEnd of input detected. Exiting...")
                return
            except Exception as analysis_error:
                print(f"Error: {analysis_error}")
                self._print_usage_help()
                attempts += 1

        if attempts >= max_attempts:
            print(f"\nMaximum attempts ({max_attempts}) reached. Exiting.")

    def _handle_pattern_analysis(self, core_pattern_input: str) -> bool:
        """Handle pattern analysis for before/after character analysis."""
        if len(core_pattern_input) != 2:
            return False

        if core_pattern_input.endswith("_") and core_pattern_input[0].isalnum():
            # X_ pattern - analyze characters that appear AFTER X
            pattern_target_char = core_pattern_input[0]
            self.analyze_character_context(pattern_target_char, "after")
            return True
        elif core_pattern_input.startswith("_") and core_pattern_input[1].isalnum():
            # _X pattern - analyze characters that appear BEFORE X
            pattern_target_char = core_pattern_input[1]
            self.analyze_character_context(pattern_target_char, "before")
            return True

        return False

    def _print_usage_help(self) -> None:
        """Print usage help message."""
        print("Please enter:")
        print("- A number (for digit analysis)")
        print("- A single letter (for alphabet analysis)")
        print("- Any 2-character combination (for 2-char combination analysis)")
        print("- Asterisks (**,***,****,etc.) for consecutive character analysis")
        print("- 'X_' (to see characters after X)")
        print("- '_X' (to see characters before X)")
        print("- 'quit' or 'exit' to quit")


def main():
    """Main function to run the analyzer with proper error handling."""
    # Configuration - Update this path as needed
    data_file_path = "shortwaves.csv"

    try:
        # Create and run the analyzer
        analyzer = CharacterAnalyzer(data_file_path)
        analyzer.run_interactive_analysis()
    except FileNotFoundError:
        print(f"Error: Could not find the file '{data_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your data file and try again.")
    finally:
        print("Analysis complete.")


if __name__ == "__main__":
    main()

# NEET INTEL / JWSCC
# HFGCS EAM CHARACTER FREQUENCY ANALYSIS SCRIPT
# PUBLIC BRANCH – v0.1 — released 250819
# IDE may throw errors; deal with it
# CONTACT 02_0279@PROTON.ME FOR ENQUIRIES
