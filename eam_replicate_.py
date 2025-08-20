import pandas as pd
import pyperclip
from datetime import datetime

def find_repeating_sets(input_strings):
    """Find repeating character sets in a list of strings."""
    repeating_sets_by_index = {}
    
    for i, input_string in enumerate(input_strings):
        # Skip None, NaN, or empty values
        if input_string is None or pd.isna(input_string) or input_string == '':
            continue
        
        # Force conversion to string
        input_string = str(input_string).strip()
        
        # Skip if too short, too long, or just 'nan'
        if len(input_string) < 4 or len(input_string) > 9000 or input_string.lower() == 'nan':
            continue
            
        repeating_sets = []
        str_len = len(input_string)
        
        for set_length in range(2, str_len // 2 + 1):
            for j in range(str_len - set_length * 2 + 1):
                set_to_check = input_string[j:j + set_length]
                if set_to_check:
                    count = input_string.count(set_to_check)
                    if count >= 2:
                        repeating_sets.append(set_to_check)
        
        if repeating_sets:
            repeating_sets_by_index[i] = repeating_sets
    
    return repeating_sets_by_index

def format_time(utc_time):
    """Format UTC time to show only hours and minutes."""
    try:
        if pd.isna(utc_time) or utc_time is None:
            return "Unknown"
        
        time_str = str(utc_time).strip()
        if time_str.lower() in ['nan', 'none', '']:
            return "Unknown"
            
        if ':' in time_str:
            time_parts = time_str.split(':')
            if len(time_parts) >= 2:
                return f"{time_parts[0]}:{time_parts[1]}"
        return time_str
    except:
        return "Unknown"

def analyze_messages(messages, source_description, utc_times=None):
    """Analyze messages for repeating sets."""
    print(f"\n=== Processing {source_description} ===")
    
    # Prepare valid data
    valid_data = []
    for i, msg in enumerate(messages):
        # Skip None, NaN, empty, or invalid messages
        if msg is None or pd.isna(msg):
            continue
            
        msg_str = str(msg).strip()
        if not msg_str or msg_str.lower() in ['nan', 'none']:
            continue
            
        time_info = "Unknown"
        if utc_times and i < len(utc_times):
            time_info = format_time(utc_times[i])
        else:
            time_info = f"line {i + 1}"
            
        valid_data.append((msg_str, time_info, i + 1))
    
    if not valid_data:
        print("No valid messages found.")
        return
    
    print(f"Found {len(valid_data)} messages to analyze.\n")
    
    # Find repeating sets
    valid_messages = [data[0] for data in valid_data]
    repeating_sets_by_index = find_repeating_sets(valid_messages)
    
    # Process results
    for i, (message, time_info, _) in enumerate(valid_data):
        if i in repeating_sets_by_index:
            time_label = "Message at"
            print(f"{time_label} {time_info}: {message}")
            print(f"Repeating sets found: {', '.join(repeating_sets_by_index[i])}")
            print()
    
    # Print summary
    if not repeating_sets_by_index:
        print("No repeating sets found.\n")

def safe_date_processing(df):
    """Safely process dates handling mixed types."""
    try:
        # Convert DATE column to string, handling NaN values
        df_copy = df.copy()
        df_copy['DATE'] = df_copy['DATE'].astype(str)
        
        # Filter out 'nan' strings and empty values
        valid_mask = ~df_copy['DATE'].isin(['nan', 'NaN', 'None', ''])
        df_filtered = df_copy[valid_mask]
        
        if df_filtered.empty:
            return None, None
            
        # Get latest date
        latest_date = df_filtered['DATE'].max()
        latest_df = df_filtered[df_filtered['DATE'] == latest_date]
        
        return latest_date, latest_df
        
    except Exception as e:
        print(f"Error in date processing: {e}")
        return None, None

def safe_date_filter(df, target_date_str):
    """Safely filter dataframe by date."""
    try:
        df_copy = df.copy()
        df_copy['DATE'] = df_copy['DATE'].astype(str)
        
        # Use string contains for more flexible matching
        mask = df_copy['DATE'].str.contains(target_date_str, na=False, regex=False)
        return df_copy[mask]
        
    except Exception as e:
        print(f"Error filtering by date: {e}")
        return pd.DataFrame()

def main():
    """Main execution function."""
    csv_file_path = 'SHORTWAVES.csv'
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path, dtype=str)
        print(f"Successfully loaded CSV with {len(df)} rows")
        
        # Check if required columns exist
        required_columns = ['DATE', 'MESSAGE', 'UTC']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns in CSV: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Safe date processing
        latest_date, latest_df = safe_date_processing(df)
        
        if latest_date is None or latest_df is None:
            print("Error: Could not process dates in CSV file")
            return
        
        messages = latest_df['MESSAGE'].tolist()
        utc_times = latest_df['UTC'].tolist()
        description = f"latest date ({latest_date}) - {len(messages)} entries"
        
        print(f"Processing messages from latest date: {latest_date}")
        analyze_messages(messages, description, utc_times)
        
        # Interactive processing loop
        while True:
            try:
                user_input = input("\nInput Y for clipboard, YYMMDD for specific date, or Enter to terminate: ").strip()
                
                if not user_input:  # Enter pressed - terminate
                    break
                elif user_input.lower() == 'y':  # Process clipboard
                    try:
                        clipboard_content = pyperclip.paste().strip()
                        if not clipboard_content:
                            print("Clipboard is empty.")
                            continue
                        
                        clipboard_lines = [line.strip() for line in clipboard_content.split('\n') if line.strip()]
                        analyze_messages(clipboard_lines, f"clipboard content - {len(clipboard_lines)} lines")
                        
                    except Exception as e:
                        print(f"Error reading clipboard: {e}")
                        
                else:  # Assume it's a date in YYMMDD format
                    try:
                        # Parse YYMMDD format and convert to YYYY.MM.DD format
                        parsed_date = datetime.strptime(user_input, '%y%m%d')
                        formatted_date = parsed_date.strftime('%Y.%m.%d')
                        
                        # Safe date filtering
                        target_df = safe_date_filter(df, formatted_date)
                        
                        if target_df.empty:
                            print(f"No data found for date: {user_input} (looking for {formatted_date})")
                            continue
                        
                        messages = target_df['MESSAGE'].tolist()
                        utc_times = target_df['UTC'].tolist()
                        description = f"date {user_input} ({formatted_date}) - {len(messages)} entries"
                        
                        print(f"Processing messages from specified date: {user_input} ({formatted_date})")
                        analyze_messages(messages, description, utc_times)
                        
                    except ValueError:
                        print(f"Invalid input: {user_input}. Please use Y for clipboard or YYMMDD for date.")
                    except Exception as e:
                        print(f"Error processing date {user_input}: {e}")
                        
            except KeyboardInterrupt:
                print("\nScript interrupted by user.")
                break
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                continue
        
        print("\nScript has been terminated.")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_file_path}")
        print("Please ensure SHORTWAVES.csv is in the same directory as this script.")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file {csv_file_path} is empty.")
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        print("Please check that the CSV file is properly formatted.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")

if __name__ == "__main__":
    main()
    
# NEET INTEL / JWSCC
# HFGCS TIMECARD GENERATOR PROGRAM
# PUBLIC BRANCH – v0.1 — released 250819
# CONTACT 02_0279@PROTON.ME FOR ENQUIRIES