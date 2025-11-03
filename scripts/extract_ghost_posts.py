#!/usr/bin/env python3
"""
Extract Ghost blog posts from the export JSON and match them to tweet dates.
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


def extract_ghost_posts(json_path):
    """Extract posts from Ghost export JSON."""
    print(f"Loading Ghost export from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Navigate to posts: data['db'][0]['data']['posts']
    posts = data['db'][0]['data']['posts']
    print(f"Found {len(posts)} total posts")
    
    return posts


def parse_post_date(published_at):
    """Parse Ghost published_at timestamp or date string."""
    if published_at is None:
        return None
    
    # Ghost timestamps can be in milliseconds
    try:
        if isinstance(published_at, (int, float)):
            # Convert milliseconds to seconds
            if published_at > 1e10:  # Likely milliseconds
                timestamp = published_at / 1000
            else:
                timestamp = published_at
            dt = datetime.fromtimestamp(timestamp)
        else:
            # Try parsing string
            dt = datetime.fromisoformat(str(published_at).replace('Z', '+00:00'))
        
        return dt
    except Exception as e:
        print(f"Warning: Could not parse date {published_at}: {e}")
        return None


def filter_posts_by_date_range(posts, start_date, end_date):
    """Filter posts within date range (inclusive)."""
    filtered = []
    
    for post in posts:
        pub_date = parse_post_date(post.get('published_at'))
        
        if pub_date is None:
            continue
        
        # Check if within range
        if start_date <= pub_date.date() <= end_date:
            filtered.append(post)
    
    return filtered


def create_posts_dataframe(posts):
    """Create DataFrame from posts for integration with tweets."""
    items = []
    
    for post in posts:
        pub_date = parse_post_date(post.get('published_at'))
        
        # Extract text content
        title = post.get('title', '')
        plaintext = post.get('plaintext', '')
        html = post.get('html', '')
        
        # Combine title and content
        text = f"{title}\n\n{plaintext}".strip()
        if not text and html:
            # Fallback to HTML if plaintext missing
            text = f"{title}\n\n{html[:500]}".strip()
        
        items.append({
            'id': f"post_{post.get('id', 'unknown')}",
            'text': text,
            'date': pub_date.isoformat() if pub_date else None,
            'date_str': pub_date.strftime('%Y-%m-%d') if pub_date else None,
            'type': 'blog_post',
            'title': title,
            'slug': post.get('slug', ''),
            'status': post.get('status', ''),
            'original_id': post.get('id', ''),
        })
    
    return pd.DataFrame(items)


def main():
    json_path = "idea-nexus-ventures.ghost.2025-11-03-16-41-30.json"
    
    # Date range for Oct 24-31, 2025
    start_date = datetime(2025, 10, 24).date()
    end_date = datetime(2025, 10, 31).date()
    
    print("=" * 80)
    print("📝 Extracting Ghost Blog Posts")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # Extract posts
    all_posts = extract_ghost_posts(json_path)
    
    # Filter by date
    filtered_posts = filter_posts_by_date_range(all_posts, start_date, end_date)
    print(f"Found {len(filtered_posts)} posts in date range")
    
    if len(filtered_posts) == 0:
        print("\n⚠️  No posts found in date range")
        # Still show some sample posts to understand the date range
        print("\nChecking sample post dates...")
        for post in all_posts[:10]:
            pub_date = parse_post_date(post.get('published_at'))
            if pub_date:
                print(f"  {pub_date.strftime('%Y-%m-%d')}: {post.get('title', 'Untitled')[:60]}")
        return
    
    # Create DataFrame
    df = create_posts_dataframe(filtered_posts)
    
    # Save to CSV
    output_file = "ghost_posts_oct24-31.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} posts to: {output_file}")
    
    # Show summary by date
    print("\n📊 Posts by Date:")
    print("-" * 80)
    date_counts = df['date_str'].value_counts().sort_index()
    for date_str, count in date_counts.items():
        titles = df[df['date_str'] == date_str]['title'].tolist()
        print(f"\n{date_str}: {count} post(s)")
        for title in titles:
            print(f"  - {title[:70]}")
    
    print("\n" + "=" * 80)
    print("✅ Extraction complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review: {output_file}")
    print(f"  2. Merge with tweets for compression analysis")
    print(f"  3. Link clusters to posts for Integration Potential (I_p)")


if __name__ == "__main__":
    main()

