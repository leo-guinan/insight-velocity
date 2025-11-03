#!/usr/bin/env python3
"""
Extract tweets for a single day and create CSV files for k-NN pipeline.
"""
import json
import sqlite3
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path


def extract_daily_tweets(db_path, username, target_date, output_items_csv="items.csv"):
    """
    Extract tweets for a single day and create items.csv for k-NN pipeline.
    
    Args:
        db_path: Path to SQLite database
        username: Username to filter (screen_name)
        target_date: Target date as datetime object (date only, time will be start of day)
        output_items_csv: Path to output items CSV file
    """
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    
    # Set date range for the single day (start of day to end of day)
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)
    
    print(f"Extracting tweets for: @{username}")
    print(f"Date: {start_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    # Convert dates to timestamps for SQLite query
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query tweets for the single day
    query = '''
        SELECT tweet_id, full_text, created_at_datetime, raw_json
        FROM tweets 
        WHERE screen_name = ? 
          AND created_at_datetime >= ? 
          AND created_at_datetime < ?
        ORDER BY created_at_datetime ASC
    '''
    
    cursor.execute(query, (username, start_timestamp, end_timestamp))
    results = cursor.fetchall()
    
    print(f"Found {len(results)} tweets")
    
    if len(results) == 0:
        print(f"No tweets found for {start_date.strftime('%Y-%m-%d')}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    
    # Build items DataFrame
    items = []
    for row in results:
        tweet_id, full_text, created_at_datetime, raw_json = row
        
        # Parse the datetime
        tweet_date = datetime.fromtimestamp(created_at_datetime, tz=timezone.utc)
        
        items.append({
            'id': f"tweet_{tweet_id}",
            'text': full_text if full_text else "",
            'date': tweet_date.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'tweet'
        })
    
    conn.close()
    
    # Create DataFrame and save
    df = pd.DataFrame(items)
    df.to_csv(output_items_csv, index=False)
    
    print(f"Created {output_items_csv} with {len(df)} items")
    print(f"Columns: {', '.join(df.columns)}")
    
    return output_items_csv


def main():
    # Configuration - extract tweets for 2025-10-31 (has 37 tweets)
    db_file = "tweets.db"
    username = "leo_guinan"
    target_date = datetime(2025, 10, 31, tzinfo=timezone.utc)
    output_items_csv = "items.csv"
    
    print("=" * 60)
    print("Daily Tweet Extractor for k-NN Pipeline")
    print("=" * 60)
    print()
    
    extract_daily_tweets(db_file, username, target_date, output_items_csv)
    
    print()
    print("=" * 60)
    print("Next step: Run k-NN pipeline")
    print("=" * 60)
    print(f"python knn_pipeline.py --items {output_items_csv} --k 5 --min_sim 0.12")


if __name__ == "__main__":
    main()

