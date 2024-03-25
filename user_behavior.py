from core import *
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns 
import numpy as np

# Examdata1
def get_user_sessions(df, session_limit=600, mean_only=False):
    """
    Analyzes user session time durations. A session is defined as a sequence of events separated by a time interval greater than the session_limit parameter. \
    The function calculates the number of sessions, mean session duration, and total duration for each user.

    Parameters:
    - df (DataFrame): Input DataFrame containing columns '_timestamp', 'user_id', and 'session_id'.
    - session_limit (int, optional): The time limit (in seconds) to define session boundaries. Defaults to 600 seconds (10 minutes).
    - mean_only (bool, optional): If True, returns only the mean session duration and total duration across all users. Defaults to False.

    Returns:
    - DataFrame or list: If mean_only is False (default), returns a DataFrame containing user-wise session statistics with columns 'user_id', 'sessions', 'session_duration', and 'total_duration'. If mean_only is True, returns a list containing the mean session duration and total duration across all users.
    """

    # Sort DataFrame by user_id, session_id, and timestamp
    df = df[['_timestamp', 'user_id', 'session_id']].sort_values(by=['user_id', 'session_id', '_timestamp']).copy()

    # Calculate duration since the previous event in seconds
    df['duration_since_previous'] = df['_timestamp'] - df['_timestamp'].shift(1)
    df['duration_since_previous'] = df['duration_since_previous'].where(df['session_id'] == df['session_id'].shift(1), pd.NaT).dt.total_seconds()

    # Determine session types
    df['session_type'] = df['duration_since_previous'].apply(lambda x: 'single_frame_session' if pd.isnull(x) else '')
    df.loc[df['duration_since_previous'] > session_limit, 'session_type'] = 'single_frame_session'
    df.loc[df['duration_since_previous'] <= session_limit, 'session_type'] = 'continued_session'

    # Set duration_since_previous to NaN for single frame sessions
    df.loc[df['duration_since_previous'] > session_limit, 'duration_since_previous'] = np.NaN

    # Differentiate new sessions from single frame sessions
    df['next_session_type'] = df['session_type'].shift(-1)
    df.loc[(df['session_type'] == 'single_frame_session') & (df['next_session_type'] == 'continued_session'), 'session_type'] = 'new_session'
    df.drop(columns=['next_session_type'], inplace=True)

    # Filter DataFrame to include only new and continued sessions
    df = df[df['session_type'].isin(['new_session', 'continued_session'])].copy()

    # Mark new sessions
    df['new_sessions'] = 0
    df.loc[(df['session_type'] == 'new_session'), 'new_sessions'] = 1

    # Calculate session start and end times
    df['session_start'] = df.groupby(['user_id', 'session_id'])['_timestamp'].transform('min')
    df['session_end'] = df.groupby(['user_id', 'session_id'])['_timestamp'].transform('max')

    # Calculate session durations
    df['session_duration'] = df['session_end'] - df['session_start']
    df['session_duration'] = df['session_duration'].dt.total_seconds()

    # Remove unnecessary columns and duplicate rows
    df = df.drop(columns=['_timestamp', 'duration_since_previous', 'session_type']).drop_duplicates()
    df = df.drop_duplicates(subset=['session_start'], keep='first')

    # Aggregate session statistics per user
    user_sessions = df.groupby('user_id').agg(sessions=('new_sessions', 'sum'), session_duration=('session_duration', 'mean'), total_duration=('session_duration','sum'))
    user_sessions = user_sessions.sort_values(by='session_duration', ascending=False)

    # Return either detailed session statistics or mean session durations and total durations
    if mean_only:
        return [user_sessions['session_duration'].mean(), user_sessions['total_duration'].mean()]
    
    return user_sessions.reset_index()

# Examdata2
def normalize_url_types(df):
    """
    This function takes a DataFrame containing data related to user activity 
    and normalizes the URL types based on certain conditions. It primarily 
    categorizes URLs into different types such as 'clip_view', 'clip_scroll', 
    'user_scroll', etc., based on various criteria.

    The normalization process filters out staging / dev URLs.

    Additionally, certain rows with specific conditions, such as 'clip_view' 
    rows without a 'share_id', are dropped to eliminate duplicates.

    Parameters:
    - df (DataFrame): Input DataFrame containing relevant data.

    Returns:
    - DataFrame: DataFrame with normalized URL types.
    """

    df = df[['name', 'user_id', '_timestamp', 'context_traits_desktop_last_launched', 'url', 'context_session_id', 'share_id']].copy()

    df['url_type'] = ''

    # Filter out staging / dev urls
    df = df[
            df['url'].str.startswith('https://allstar') | 
            df['url'].str.startswith('file:/')
            ].reset_index(drop=True)

    # There is a way to parallelize this with dask, but I can't remember atm
    for key, value in web_mapper.items():
        df.loc[(df['url'].str.contains(key, regex=False)), 'url_type'] = value

    for key, value in desktop_mapper.items():
        df.loc[(df['name'] == key), 'url_type'] = value

    # Niche Cases
    df.loc[(
        (df['url'] == 'https://allstar.gg/clips') | 
        (df['url'] == 'https://allstar.gg/clip') |
        (df['url'] == 'https://allstar.gg/')
        ), 'url_type'] = 'clip_scroll'

    # Desktop app urls with no name are impossible to differentiate, so I am just going to mark them as unknown
    df.loc[(df['url'].str.contains('file:/') & (df['name'].isna())), 'url_type'] = 'unknown'

    # The rest of the urls are all user landing pages but with no /u/ tag so impossible to differentiate
    # Best solution was to clean literally everything else so I could do this
    df.loc[(df['url'] == ''), 'url_type'] = 'user_scroll'

    df = df.reset_index(drop=True)
    
    # Drop clip_view rows with no share_id, as they act as duplicate clip_view events
    df = df.drop(df[(df['url_type'] == 'clip_view') & (df['share_id'].isna())].index).reset_index(drop=True)

    return df

def get_session_summary_df(df, filtered=True):
    """
    This function takes a DataFrame containing normalized URL types and generates 
    a summary DataFrame of user session activity. It groups the data by 'user_id' 
    and 'context_session_id', and then calculates the counts of each 'url_type'.

    Parameters:
    - df (DataFrame): Input DataFrame containing normalized URL types.
    - filtered (bool): Flag indicating whether to filter columns based on 
                       filtered_values in `core.py`. Defaults to True.

    Returns:
    - DataFrame: Summary DataFrame containing user session activity.
    """

    df = normalize_url_types(df)

    # Get the value counts of url_types and then unstack them for easy manipulation
    df = df.groupby(['user_id', 'context_session_id'])['url_type'].value_counts()
    df = df.unstack().fillna(0).reset_index()

    # Pages load 10 clips at a time and count as 1 scroll
    # Not sure how to handle un-even scroll counts
    # Also haven't decided how to handle landing page +1 
    # For now I am just rounding up 
    df['user_scroll'] = np.ceil((df['user_scroll'] / 10))
    df['clip_scroll'] = np.ceil((df['clip_scroll'] / 10))

    # Filter out columns that are not in the list of filtered_values
    if filtered:
        cols = ['user_id', 'context_session_id']
        cols.extend([x for x in filtered_values if x in list(df.columns)])
        df = df[cols].copy()

    return df

def get_user_behavior_barh_chart(df, filtered=True, mean=False):
    """
    This function generates a horizontal bar chart representing user behavior 
    based on the provided DataFrame containing user session summary. It calls 
    'get_session_summary_df' to prepare the data and then calculates either 
    total counts or mean values for each type of URL activity.

    The bar chart visualizes user behavior by displaying the counts or mean 
    values for different types of URL activities, such as 'user_scroll', 
    'clip_scroll', 'clip_view', etc. 

    Parameters:
    - df (DataFrame): Input DataFrame containing user session summary.
    - filtered (bool): Flag indicating whether to filter columns based on 
                       predefined criteria. Defaults to True.
    - mean (bool): Flag indicating whether to calculate mean values instead 
                   of total counts. Defaults to False.

    Returns:
    - None
    """

    df = get_session_summary_df(df, filtered=filtered)

    if mean:
        url_type_counts = df.iloc[:, 2:].mean().sort_values(ascending=False)
    else:
        url_type_counts = df.iloc[:, 2:].sum().sort_values(ascending=False)

    background='#0d1a28'
    facecolor='#0a1521'
    plt.figure(figsize=(10,5)).set_facecolor(background)

    # Setting some simple plt params

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("cyberpunk")
    plt.grid(c='#0d1a28')
    plt.setp(ax.spines.values(), visible=False)
    plt.tick_params(axis='both', colors='w')
    
    fig.set_facecolor('#0d1a28')
    ax.set_facecolor('#0a1521')
    ax.set_axisbelow(True)

    sns.barplot(x=url_type_counts.values, y=url_type_counts.index, alpha=1, orient='h', palette='crest', legend=False).set_facecolor(facecolor)
    
    plt.ylabel(None)
    plt.xlabel(None)
