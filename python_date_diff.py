import pandas as pd
import numpy as np

# 2026 US Federal Holidays
US_HOLIDAYS_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Washington's Birthday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day observed
    "2026-09-07",  # Labor Day
    "2026-10-12",  # Columbus Day
    "2026-11-11",  # Veterans Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25"   # Christmas Day
]

US_HOLIDAYS_2026 = np.array(
    US_HOLIDAYS_2026,
    dtype="datetime64[D]"
)


def working_day_difference(transaction_date, clearing_date):

    transaction_date = np.datetime64(transaction_date, "D")
    clearing_date = np.datetime64(clearing_date, "D")

    if clearing_date < transaction_date:
        return -1

    # Monday-Saturday = working
    # Sunday = non-working
    working_days = np.busday_count(
        transaction_date,
        clearing_date,
        weekmask="1111110",
        holidays=US_HOLIDAYS_2026
    )
df["working_day_diff"] = df.apply(
    lambda x: working_day_difference(
        x["transaction_date"],
        x["clearing_date"]
    ),
    axis=1
)

df["clearing_flag"] = np.where(
    df["working_day_diff"].between(0, 2),
    "Yes",
    "No"
)
    return working_days



"""
US National Holidays 2026 - Python Implementation
Transaction Clearing Date Calculation with Business Days

Author: Data Analytics Team
Date: 2026
Version: 1.0
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

# =====================================================
# US HOLIDAYS 2026
# =====================================================

US_HOLIDAYS_2026 = {
    'New Year\'s Day': '2026-01-01',
    'MLK Jr. Day': '2026-01-19',
    'Presidents Day': '2026-02-16',
    'Good Friday': '2026-04-03',
    'Memorial Day': '2026-05-25',
    'Juneteenth': '2026-06-19',
    'Independence Day': '2026-07-04',
    'Labor Day': '2026-09-07',
    'Columbus Day': '2026-10-12',
    'Veterans Day': '2026-11-11',
    'Thanksgiving': '2026-11-26',
    'Christmas': '2026-12-25',
}

# Alternative format: list of tuples
HOLIDAYS_2026_LIST = [
    ('2026-01-01', 'New Year\'s Day'),
    ('2026-01-19', 'MLK Jr. Day'),
    ('2026-02-16', 'Presidents Day'),
    ('2026-04-03', 'Good Friday'),
    ('2026-05-25', 'Memorial Day'),
    ('2026-06-19', 'Juneteenth'),
    ('2026-07-04', 'Independence Day'),
    ('2026-09-07', 'Labor Day'),
    ('2026-10-12', 'Columbus Day'),
    ('2026-11-11', 'Veterans Day'),
    ('2026-11-26', 'Thanksgiving'),
    ('2026-12-25', 'Christmas'),
]

# Convert to pandas DatetimeIndex for efficient operations
HOLIDAYS_2026_DATES = pd.to_datetime(list(US_HOLIDAYS_2026.values()))

# =====================================================
# CREATE HOLIDAYS DATAFRAME
# =====================================================

def create_holidays_dataframe() -> pd.DataFrame:
    """Create a DataFrame with 2026 US holidays and detailed information."""
    df = pd.DataFrame(HOLIDAYS_2026_LIST, columns=['holiday_date', 'holiday_name'])
    df['holiday_date'] = pd.to_datetime(df['holiday_date'])
    df['day_name'] = df['holiday_date'].dt.day_name()
    df['day_of_week'] = df['holiday_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['holiday_date'].dt.month
    df['date_str'] = df['holiday_date'].dt.strftime('%Y-%m-%d')
    
    return df

# =====================================================
# FUNCTION 1: BASIC BUSINESS DAY CALCULATION
# =====================================================

def is_working_day(date: datetime, holidays: List[datetime] = None) -> bool:
    """
    Check if a date is a working day.
    
    Working days: Monday-Saturday (NOT Sunday)
    Non-working days: Sunday + Holidays
    
    Args:
        date: The date to check
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        True if working day, False otherwise
    """
    if holidays is None:
        holidays = HOLIDAYS_2026_DATES
    
    # 6 = Sunday (not a working day)
    if date.weekday() == 6:
        return False
    
    # Check if date is a holiday
    date_only = date.date() if isinstance(date, datetime) else date
    return date_only not in [h.date() if isinstance(h, datetime) else h for h in holidays]


def calculate_business_days(
    start_date: datetime, 
    end_date: datetime,
    holidays: List[datetime] = None
) -> int:
    """
    Calculate the number of business days between two dates.
    
    Business days include Monday through Saturday (excluding Sundays and holidays).
    
    Args:
        start_date: Transaction date (exclusive)
        end_date: Clearing date (inclusive)
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        Number of business days elapsed
    
    Example:
        >>> start = datetime(2026, 11, 24)  # Tuesday
        >>> end = datetime(2026, 11, 30)    # Monday
        >>> calculate_business_days(start, end)
        4
    """
    if holidays is None:
        holidays = HOLIDAYS_2026_DATES
    
    business_days = 0
    current_date = start_date + timedelta(days=1)
    
    while current_date <= end_date:
        if is_working_day(current_date, holidays):
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days


def is_cleared_within_2_days(
    transaction_date: datetime,
    clearing_date: datetime,
    holidays: List[datetime] = None
) -> str:
    """
    Determine if a transaction cleared within 2 business days.
    
    Args:
        transaction_date: When the transaction occurred
        clearing_date: When the transaction was cleared
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        'Yes' if cleared within 2 business days, 'No' otherwise
    """
    business_days = calculate_business_days(transaction_date, clearing_date, holidays)
    return 'Yes' if business_days <= 2 else 'No'


# =====================================================
# FUNCTION 2: PANDAS OPTIMIZED CALCULATION
# =====================================================

def calculate_business_days_pandas(
    df: pd.DataFrame,
    start_col: str = 'transaction_date',
    end_col: str = 'clearing_date',
    holidays: List[datetime] = None
) -> pd.DataFrame:
    """
    Vectorized calculation of business days for a DataFrame.
    
    Much faster than loop-based approach for large datasets.
    
    Args:
        df: DataFrame with transaction and clearing dates
        start_col: Column name for transaction dates
        end_col: Column name for clearing dates
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        DataFrame with additional columns for business days and clearing status
    
    Example:
        >>> df = pd.DataFrame({
        ...     'transaction_id': ['TXN001', 'TXN002'],
        ...     'transaction_date': ['2026-11-27', '2026-11-24'],
        ...     'clearing_date': ['2026-11-29', '2026-11-30']
        ... })
        >>> result = calculate_business_days_pandas(df)
    """
    if holidays is None:
        holidays = HOLIDAYS_2026_DATES
    
    df_copy = df.copy()
    df_copy[start_col] = pd.to_datetime(df_copy[start_col])
    df_copy[end_col] = pd.to_datetime(df_copy[end_col])
    
    # Calculate business days for each row
    def count_business_days(row):
        start = row[start_col]
        end = row[end_col]
        count = 0
        current = start + timedelta(days=1)
        
        while current <= end:
            if is_working_day(current, holidays):
                count += 1
            current += timedelta(days=1)
        
        return count
    
    df_copy['business_days_elapsed'] = df_copy.apply(count_business_days, axis=1)
    df_copy['is_cleared_within_2_days'] = df_copy['business_days_elapsed'].apply(
        lambda x: 'Yes' if x <= 2 else 'No'
    )
    
    return df_copy


# =====================================================
# FUNCTION 3: EFFICIENT DATE DIMENSION APPROACH
# =====================================================

def create_date_dimension(
    start_date: str = '2026-01-01',
    end_date: str = '2026-12-31',
    holidays: List[datetime] = None
) -> pd.DataFrame:
    """
    Create a date dimension table for efficient lookups.
    
    Pre-calculates whether each date is a working day.
    Use this for better performance on large datasets.
    
    Args:
        start_date: Starting date (YYYY-MM-DD format)
        end_date: Ending date (YYYY-MM-DD format)
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        DataFrame with date information and working day flag
    """
    if holidays is None:
        holidays = HOLIDAYS_2026_DATES
    
    # Generate all dates in range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_dates = pd.DataFrame({
        'date_value': date_range,
        'day_name': date_range.strftime('%A'),
        'day_of_week': date_range.dayofweek,
        'month': date_range.month,
        'year': date_range.year,
    })
    
    # Mark holidays
    holiday_dates = [h.date() if isinstance(h, datetime) else h for h in holidays]
    df_dates['is_holiday'] = df_dates['date_value'].dt.date.isin(holiday_dates)
    
    # Mark Sundays
    df_dates['is_sunday'] = df_dates['day_of_week'] == 6
    
    # Mark working days (not Sunday and not holiday)
    df_dates['is_working_day'] = ~(df_dates['is_sunday'] | df_dates['is_holiday'])
    df_dates['is_non_working_day'] = ~df_dates['is_working_day']
    
    return df_dates


def calculate_business_days_with_dimension(
    transaction_date: datetime,
    clearing_date: datetime,
    date_dimension: pd.DataFrame
) -> int:
    """
    Calculate business days using pre-computed date dimension.
    
    Much faster for repeated calculations on same date range.
    
    Args:
        transaction_date: Start date
        clearing_date: End date
        date_dimension: Pre-computed date dimension DataFrame
    
    Returns:
        Number of business days
    """
    mask = (
        (date_dimension['date_value'].dt.date > transaction_date.date()) &
        (date_dimension['date_value'].dt.date <= clearing_date.date()) &
        (date_dimension['is_working_day'] == True)
    )
    return len(date_dimension[mask])


# =====================================================
# FUNCTION 4: DETAILED ANALYSIS WITH BREAKDOWN
# =====================================================

def analyze_clearing_transaction(
    transaction_date: datetime,
    clearing_date: datetime,
    holidays_dict: Dict = None,
    holidays_list: List[datetime] = None
) -> Dict:
    """
    Detailed analysis of a clearing transaction.
    
    Provides breakdown of which dates are counted and why.
    
    Args:
        transaction_date: When transaction occurred
        clearing_date: When transaction cleared
        holidays_dict: Dictionary mapping dates to holiday names
        holidays_list: List of holiday dates
    
    Returns:
        Dictionary with detailed analysis
    """
    if holidays_dict is None:
        holidays_dict = US_HOLIDAYS_2026
    if holidays_list is None:
        holidays_list = HOLIDAYS_2026_DATES
    
    # Convert to datetime if needed
    if isinstance(transaction_date, str):
        transaction_date = pd.to_datetime(transaction_date)
    if isinstance(clearing_date, str):
        clearing_date = pd.to_datetime(clearing_date)
    
    total_days = (clearing_date - transaction_date).days
    business_days = 0
    sundays_count = 0
    holidays_found = []
    working_days_list = []
    
    current_date = transaction_date + timedelta(days=1)
    
    while current_date <= clearing_date:
        day_name = current_date.strftime('%A')
        date_str = current_date.strftime('%Y-%m-%d')
        
        if current_date.weekday() == 6:  # Sunday
            sundays_count += 1
        elif date_str in [pd.to_datetime(d).strftime('%Y-%m-%d') for d in holidays_dict.values()]:
            # Check if it's a holiday
            for holiday_name, holiday_date in holidays_dict.items():
                if pd.to_datetime(holiday_date).strftime('%Y-%m-%d') == date_str:
                    holidays_found.append((date_str, holiday_name, day_name))
                    break
        else:
            business_days += 1
            working_days_list.append((date_str, day_name))
        
        current_date += timedelta(days=1)
    
    return {
        'transaction_date': transaction_date.strftime('%Y-%m-%d'),
        'clearing_date': clearing_date.strftime('%Y-%m-%d'),
        'total_calendar_days': total_days,
        'sundays_excluded': sundays_count,
        'holidays_excluded': holidays_found,
        'business_days_elapsed': business_days,
        'working_days': working_days_list,
        'is_cleared_within_2_days': 'Yes' if business_days <= 2 else 'No',
        'clearing_status': (
            'Same Day' if business_days == 0
            else 'Next Business Day' if business_days == 1
            else 'Within 2 Business Days' if business_days == 2
            else 'Beyond 2 Business Days'
        )
    }


# =====================================================
# FUNCTION 5: BATCH PROCESSING
# =====================================================

def process_transactions_batch(
    transactions: List[Tuple[str, str, str]],
    holidays: List[datetime] = None
) -> pd.DataFrame:
    """
    Process a batch of transactions.
    
    Args:
        transactions: List of tuples (transaction_id, transaction_date, clearing_date)
        holidays: List of holiday dates (default: 2026 holidays)
    
    Returns:
        DataFrame with clearing status for all transactions
    
    Example:
        >>> txns = [
        ...     ('TXN001', '2026-11-27', '2026-11-29'),
        ...     ('TXN002', '2026-11-24', '2026-11-30'),
        ... ]
        >>> result = process_transactions_batch(txns)
    """
    if holidays is None:
        holidays = HOLIDAYS_2026_DATES
    
    results = []
    
    for txn_id, txn_date, clear_date in transactions:
        txn_dt = pd.to_datetime(txn_date)
        clear_dt = pd.to_datetime(clear_date)
        
        business_days = calculate_business_days(txn_dt, clear_dt, holidays)
        status = 'Yes' if business_days <= 2 else 'No'
        
        results.append({
            'transaction_id': txn_id,
            'transaction_date': txn_date,
            'clearing_date': clear_date,
            'business_days': business_days,
            'is_cleared_within_2_days': status
        })
    
    return pd.DataFrame(results)


# =====================================================
# SAMPLE TEST DATA
# =====================================================

SAMPLE_TRANSACTIONS = [
    ('TXN_001', '2026-11-27', '2026-11-29'),  # Wed to Fri (before Thanksgiving)
    ('TXN_002', '2026-11-24', '2026-11-30'),  # Tue to Mon (Thanksgiving week)
    ('TXN_003', '2026-11-27', '2026-12-02'),  # Wed to Wed (crosses Thanksgiving)
    ('TXN_004', '2026-07-02', '2026-07-06'),  # Thu to Mon (Independence Day)
    ('TXN_005', '2026-01-02', '2026-01-06'),  # Fri to Tue (New Year)
    ('TXN_006', '2026-12-24', '2026-12-28'),  # Christmas week
    ('TXN_007', '2026-05-22', '2026-05-26'),  # Memorial Day
    ('TXN_008', '2026-11-29', '2026-11-30'),  # Fri to Sat (post-Thanksgiving)
]


# =====================================================
# MAIN EXECUTION & EXAMPLES
# =====================================================

def main():
    """Run examples and demonstrations."""
    
    print("=" * 70)
    print("US NATIONAL HOLIDAYS 2026 - PYTHON IMPLEMENTATION")
    print("=" * 70)
    
    # ===== EXAMPLE 1: Display holidays =====
    print("\n1. US HOLIDAYS 2026:")
    print("-" * 70)
    holidays_df = create_holidays_dataframe()
    print(holidays_df.to_string(index=False))
    print(f"\nTotal holidays: {len(holidays_df)}")
    
    # ===== EXAMPLE 2: Single transaction analysis =====
    print("\n\n2. SINGLE TRANSACTION EXAMPLE:")
    print("-" * 70)
    analysis = analyze_clearing_transaction(
        datetime(2026, 11, 24),  # Tuesday
        datetime(2026, 11, 30)   # Monday
    )
    print(f"Transaction Date: {analysis['transaction_date']}")
    print(f"Clearing Date: {analysis['clearing_date']}")
    print(f"Total Calendar Days: {analysis['total_calendar_days']}")
    print(f"Sundays Excluded: {analysis['sundays_excluded']}")
    print(f"Holidays Excluded: {analysis['holidays_excluded']}")
    print(f"Business Days Elapsed: {analysis['business_days_elapsed']}")
    print(f"Cleared within 2 business days: {analysis['is_cleared_within_2_days']} ✓")
    print(f"Status: {analysis['clearing_status']}")
    
    # ===== EXAMPLE 3: Batch processing =====
    print("\n\n3. BATCH PROCESSING - SAMPLE TRANSACTIONS:")
    print("-" * 70)
    results_df = process_transactions_batch(SAMPLE_TRANSACTIONS)
    print(results_df.to_string(index=False))
    
    # Summary
    cleared_yes = len(results_df[results_df['is_cleared_within_2_days'] == 'Yes'])
    cleared_no = len(results_df[results_df['is_cleared_within_2_days'] == 'No'])
    print(f"\nSummary:")
    print(f"  Cleared within 2 days: {cleared_yes}")
    print(f"  NOT cleared within 2 days: {cleared_no}")
    
    # ===== EXAMPLE 4: Using pandas DataFrame =====
    print("\n\n4. WORKING WITH PANDAS DATAFRAME:")
    print("-" * 70)
    df_transactions = pd.DataFrame(
        SAMPLE_TRANSACTIONS,
        columns=['transaction_id', 'transaction_date', 'clearing_date']
    )
    df_result = calculate_business_days_pandas(df_transactions)
    print(df_result[['transaction_id', 'transaction_date', 'clearing_date', 
                     'business_days_elapsed', 'is_cleared_within_2_days']].to_string(index=False))
    
    # ===== EXAMPLE 5: Date Dimension Table =====
    print("\n\n5. DATE DIMENSION TABLE (2026-11 NOVEMBER SAMPLE):")
    print("-" * 70)
    date_dim = create_date_dimension('2026-11-01', '2026-11-30')
    print(date_dim[['date_value', 'day_name', 'is_holiday', 'is_sunday', 'is_working_day']]
          .head(30).to_string(index=False))
    
    # ===== EXAMPLE 6: Holiday statistics =====
    print("\n\n6. HOLIDAY STATISTICS FOR 2026:")
    print("-" * 70)
    holiday_stats = holidays_df.groupby('day_name').size().sort_values(ascending=False)
    print("Holidays by Day of Week:")
    print(holiday_stats)
    
    holiday_by_month = holidays_df.groupby('month').size().sort_values(ascending=False)
    print("\nHolidays by Month:")
    for month, count in holiday_by_month.items():
        print(f"  Month {month:2d}: {count} holiday(s)")
    
    # ===== EXAMPLE 7: Performance comparison =====
    print("\n\n7. PERFORMANCE COMPARISON:")
    print("-" * 70)
    import time
    
    large_batch = [(f'TXN{i:05d}', 
                   (datetime(2026, 1, 1) + timedelta(days=i*10)).strftime('%Y-%m-%d'),
                   (datetime(2026, 1, 1) + timedelta(days=i*10+5)).strftime('%Y-%m-%d'))
                   for i in range(1000)]
    
    # Method 1: Loop-based
    start = time.time()
    result1 = process_transactions_batch(large_batch)
    loop_time = time.time() - start
    
    # Method 2: Pandas-based
    df_large = pd.DataFrame(large_batch, columns=['transaction_id', 'transaction_date', 'clearing_date'])
    start = time.time()
    result2 = calculate_business_days_pandas(df_large)
    pandas_time = time.time() - start
    
    print(f"Loop-based processing (1000 rows): {loop_time:.4f} seconds")
    print(f"Pandas-based processing (1000 rows): {pandas_time:.4f} seconds")
    print(f"Speedup: {loop_time/pandas_time:.2f}x faster with pandas")
    
    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()


# =====================================================
# QUICK REFERENCE USAGE
# =====================================================

"""
QUICK START GUIDE:

1. Single transaction:
   >>> from datetime import datetime
   >>> result = is_cleared_within_2_days(
   ...     datetime(2026, 11, 27),
   ...     datetime(2026, 11, 29)
   ... )
   >>> print(result)  # Output: 'Yes'

2. Calculate business days:
   >>> business_days = calculate_business_days(
   ...     datetime(2026, 11, 24),
   ...     datetime(2026, 11, 30)
   ... )
   >>> print(business_days)  # Output: 4

3. Detailed analysis:
   >>> analysis = analyze_clearing_transaction(
   ...     '2026-11-24',
   ...     '2026-11-30'
   ... )
   >>> print(analysis['clearing_status'])

4. Batch processing with DataFrame:
   >>> df = pd.DataFrame({
   ...     'transaction_id': ['TXN001', 'TXN002'],
   ...     'transaction_date': ['2026-11-27', '2026-11-24'],
   ...     'clearing_date': ['2026-11-29', '2026-11-30']
   ... })
   >>> result = calculate_business_days_pandas(df)

5. Using date dimension (best for large datasets):
   >>> date_dim = create_date_dimension()
   >>> business_days = calculate_business_days_with_dimension(
   ...     datetime(2026, 11, 27),
   ...     datetime(2026, 11, 29),
   ...     date_dim
   ... )

6. Get holidays dataframe:
   >>> holidays = create_holidays_dataframe()
   >>> print(holidays)
"""
