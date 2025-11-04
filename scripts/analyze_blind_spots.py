#!/usr/bin/env python3
"""
Analyze blind spots to find:
1. What private content (AI archive) triggered the blind spot
2. What public clusters it should align with
3. What public content (tweets) exist in those aligned clusters
4. Auto-generate content suggestions (tweet hooks, thread outlines, blog skeletons)
"""
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent))
from generate_blind_spot_content import generate_content_suggestions

def load_ai_archive_content(data_room_path, date_str):
    """Load AI archive content for a specific date."""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    day = date_obj.strftime('%d')
    
    # Try different possible locations
    possible_dirs = [
        Path(data_room_path) / year / month / day / 'ai_archives',
        Path(data_room_path) / 'ai_archives' / year / month / day,
        Path(data_room_path) / 'ai_archives' / 'claude' / year / month / day,
        Path(data_room_path) / 'ai_archives' / 'anthropic' / year / month / day,
    ]
    
    items = []
    for archive_dir in possible_dirs:
        if archive_dir.exists():
            for json_file in archive_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Extract conversation text
                            text_parts = []
                            if 'messages' in data:
                                for msg in data.get('messages', []):
                                    if isinstance(msg, dict):
                                        content = msg.get('content', '')
                                        if isinstance(content, list):
                                            content = ' '.join(str(c) for c in content if isinstance(c, (str, dict)))
                                        if content:
                                            text_parts.append(str(content))
                            if not text_parts and 'text' in data:
                                text_parts.append(str(data['text']))
                            
                            if text_parts:
                                # Clean up Claude export artifacts
                                full_text = ' '.join(text_parts)
                                # Remove common Claude UI messages
                                full_text = full_text.replace('This block is not supported on your current device yet.', '')
                                full_text = full_text.replace('```', '')  # Remove code block markers
                                # Collapse multiple whitespace
                                full_text = re.sub(r'\s+', ' ', full_text).strip()
                                
                                items.append({
                                    'file': json_file.name,
                                    'text': full_text[:500],  # First 500 chars
                                    'full_text': full_text
                                })
                except Exception as e:
                    continue
            if items:
                break  # Found items, don't check other locations
    
    return items

def load_tweet_content(data_room_path, tweet_id, date_str=None):
    """Load tweet content by ID from data room. Searches broadly if date not found."""
    # Extract numeric ID from tweet_ID format
    numeric_id = tweet_id.replace('tweet_', '')
    
    # First try the specific date if provided
    if date_str:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        possible_dirs = [
            Path(data_room_path) / 'tweets' / 'ideas' / year / month / day,
            Path(data_room_path) / 'tweets' / 'conversations' / year / month / day,
        ]
        
        for tweet_dir in possible_dirs:
            if not tweet_dir.exists():
                continue
            
            for json_file in tweet_dir.glob('*.json'):
                result = _try_load_tweet_file(json_file, numeric_id, tweet_id)
                if result:
                    return result
    
    # If not found on specific date, search more broadly in tweets directory
    tweets_base = Path(data_room_path) / 'tweets'
    if tweets_base.exists():
        for json_file in tweets_base.rglob('*.json'):
            result = _try_load_tweet_file(json_file, numeric_id, tweet_id)
            if result:
                return result
    
    return None

def _try_load_tweet_file(json_file, numeric_id, tweet_id):
    """Try to load a tweet from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Check if this is the tweet we're looking for
            tweet_data_id = str(data.get('id_str', data.get('id', '')))
            if tweet_data_id == numeric_id:
                text = data.get('text', data.get('full_text', ''))
                if text:
                    return {
                        'id': tweet_id,
                        'text': text,
                        'file': str(json_file.relative_to(json_file.parents[3]))  # Show relative path
                    }
    except:
        pass
    return None

def find_public_cluster_examples(public_items_df, cluster_id, limit=5):
    """Find example items from a public cluster."""
    # public_clusters.csv has: id, label, probability
    # label is the cluster ID
    cluster_items = public_items_df[public_items_df['label'] == cluster_id]
    if cluster_items.empty:
        return []
    
    examples = []
    for _, row in cluster_items.head(limit).iterrows():
        # We don't have text in clusters CSV, just IDs
        # Would need to match back to original items
        item_id = row.get('id', '')
        examples.append({
            'id': item_id,
            'probability': row.get('probability', 0)
        })
    return examples

def analyze_blind_spots(two_pole_output_dir, data_room_path, top_n=10):
    """Analyze top blind spots and show what they are."""
    output_dir = Path(two_pole_output_dir)
    data_room = Path(data_room_path)
    
    # Load data
    print("📥 Loading blind spot data...")
    blind_spots_df = pd.read_csv(output_dir / 'time_aware_blind_spots.csv')
    blind_spots_df['date'] = pd.to_datetime(blind_spots_df['date'])
    
    private_clusters_df = pd.read_csv(output_dir / 'private_clusters.csv')
    
    # Load lead_lag_map to find aligned public clusters
    lead_lag_map_df = pd.read_csv(output_dir / 'lead_lag_map.csv')
    
    # Load public items if available (from original data)
    public_items_df = None
    public_clusters_file = output_dir / 'public_clusters.csv'
    if public_clusters_file.exists():
        public_items_df = pd.read_csv(public_clusters_file)
    
    # Get top N blind spots
    top_blind_spots = blind_spots_df.nlargest(top_n, 'blind_spot_score')
    
    print(f"\n{'='*80}")
    print(f"🔍 Analyzing Top {top_n} Blind Spots")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, row in top_blind_spots.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        private_cluster = row['private_cluster']
        score = row['blind_spot_score']
        activation = row['private_activation']
        expected_lag = row['expected_lag']
        
        print(f"{'='*80}")
        print(f"📍 Blind Spot #{idx+1}: {date_str}, Private Cluster {private_cluster}, Score={score:.2f}")
        print(f"{'='*80}")
        
        # Show categorization
        blind_spot_type = row.get('blind_spot_type', 'unknown')
        is_reactivation_gap = row.get('is_reactivation_gap', False)
        max_similarity = row.get('max_similarity', 0.0)
        
        type_label = {
            'reactivation_gap': 'Type A: Reactivation Gap (Dormant Public Analogue)',
            'burst_no_collapse': 'Type B: Burst→No-Collapse',
            'channel_mismatch': 'Type C: Channel Mismatch (Research-Dense → Lightweight)',
            'new_topic': 'New Topic (No Historical Precedent)',
            'insufficient_activity': 'Insufficient Public Activity'
        }.get(blind_spot_type, f'Type: {blind_spot_type}')
        
        print(f"   Category: {type_label}")
        if is_reactivation_gap:
            print(f"   ⚠️  Reactivation Gap: No recent public matches (±30 days), only historical")
        else:
            print(f"   Expected lag: {expected_lag:.1f} days" if pd.notna(expected_lag) else "   Expected lag: N/A")
        print(f"   Private activation: {activation}")
        print(f"   Max similarity to public clusters: {max_similarity:.3f}")
        if pd.notna(row.get('recent_public_count')):
            print(f"   Recent public activity (±30d): {int(row.get('recent_public_count', 0))} items")
            print(f"   Historical public activity: {int(row.get('historical_public_count', 0))} items")
        print()
        
        # Find private cluster items for this date
        print("🔒 PRIVATE CONTENT (AI Archive):")
        print("-" * 80)
        
        # Find items in this private cluster
        # private_clusters_df has: id, label, probability
        # label is the cluster ID (private_cluster)
        cluster_items = private_clusters_df[private_clusters_df['label'] == private_cluster]
        
        # Load AI archive content for this date
        archive_content = load_ai_archive_content(data_room, date_str)
        
        if cluster_items.empty:
            if archive_content:
                print(f"   Found {len(archive_content)} AI archive files on {date_str}")
                print(f"   (Cluster label {private_cluster} might be an outlier or unmatched)")
                for i, item in enumerate(archive_content[:3], 1):
                    print(f"\n   [{i}] {item['file']}")
                    print(f"   {item['text']}...")
            else:
                print(f"   ⚠️  No AI archive files found on {date_str} in cluster {private_cluster}")
        else:
            # Match cluster item IDs to archive files
            cluster_ids = set(cluster_items['id'].tolist())
            matched_items = []
            unmatched_items = []
            
            for arch in archive_content:
                # Try to match ID pattern in filename
                # ID format: ai_anthropic_claude_XXXXX
                # File might be: claude_XXXXX.json or similar
                file_match = False
                for cid in cluster_ids:
                    # Extract hash from ID (last part after last underscore)
                    if '_' in cid:
                        hash_part = cid.split('_')[-1]
                        if hash_part.lower() in arch['file'].lower():
                            matched_items.append((cid, arch))
                            file_match = True
                            break
                
                if not file_match:
                    unmatched_items.append(arch)
            
            if matched_items:
                print(f"   Found {len(matched_items)} items in cluster {private_cluster} on {date_str}")
                for i, (cid, item) in enumerate(matched_items[:3], 1):
                    print(f"\n   [{i}] {item['file']} (ID: {cid})")
                    print(f"   {item['text']}...")
            
            if not matched_items and archive_content:
                # Fallback: show all archives for this date
                print(f"   Showing all {len(archive_content)} AI archive files on {date_str}:")
                for i, item in enumerate(archive_content[:3], 1):
                    print(f"\n   [{i}] {item['file']}")
                    print(f"   {item['text']}...")
        
        print()
        
        # Find aligned public clusters
        print("🔗 ALIGNED PUBLIC CLUSTERS (Expected Output):")
        print("-" * 80)
        
        aligned = lead_lag_map_df[lead_lag_map_df['private_cluster'] == private_cluster]
        if aligned.empty:
            print(f"   ⚠️  No aligned public clusters found for private cluster {private_cluster}")
        else:
            aligned = aligned.sort_values('cosine_similarity', ascending=False).head(3)
            for i, (_, align_row) in enumerate(aligned.iterrows(), 1):
                pub_cluster = align_row['public_cluster']
                similarity = align_row.get('cosine_similarity', 0)
                optimal_lag = align_row.get('optimal_lag', 0)
                
                print(f"\n   [{i}] Public Cluster {pub_cluster}")
                print(f"       Similarity: {similarity:.3f}")
                print(f"       Optimal lag: {optimal_lag:.1f} days")
                
                # Find example public items from this cluster
                if public_items_df is not None:
                    examples = find_public_cluster_examples(public_items_df, pub_cluster, limit=3)
                    if examples:
                        print(f"       Found {len(examples)} items in this cluster:")
                        for ex in examples[:2]:
                            item_id = ex.get('id', 'unknown')
                            # Try to load tweet content
                            tweet_content = load_tweet_content(data_room, item_id, date_str)
                            if tweet_content:
                                print(f"\n         [{item_id}]")
                                print(f"         {tweet_content['text'][:300]}...")
                            else:
                                print(f"         • {item_id} (prob: {ex.get('probability', 0):.3f})")
                                print(f"           (Could not load tweet content from data room)")
                else:
                    print(f"       (Use public_clusters.csv to find item IDs, then match to original tweets)")
        
        print()
        
        # Store result
        results.append({
            'date': date_str,
            'private_cluster': private_cluster,
            'score': score,
            'private_items_count': len(cluster_items) if not cluster_items.empty else 0,
            'aligned_public_clusters': aligned['public_cluster'].tolist() if not aligned.empty else []
        })
        
        # Generate content suggestions
        print("💡 CONTENT SUGGESTIONS:")
        print("-" * 80)
        
        # Extract a summary of private content for context
        private_summary = ""
        if archive_content:
            # Use first few words from first archive
            first_text = archive_content[0].get('text', '')[:100]
            private_summary = first_text.split('.')[0] if first_text else ""
        
        # Convert row to dict for content generation
        row_dict = row.to_dict()
        suggestions = generate_content_suggestions(row_dict, private_summary)
        
        print(f"\n   Tweet Hook:")
        print(f"   \"{suggestions['tweet_hook']}\"")
        
        print(f"\n   Recommended Format:")
        print(f"   {suggestions['recommended_format']}")
        
        print(f"\n   Thread Outline (5 bullets):")
        for bullet in suggestions['thread_outline']:
            print(f"   {bullet}")
        
        print(f"\n   Action Items:")
        for action in suggestions['action_items']:
            print(f"   • {action}")
        
        print(f"\n   Blog Outline ({suggestions['blog_outline']['target_word_count']} words):")
        print(f"   Title: {suggestions['blog_outline']['title']}")
        for section in suggestions['blog_outline']['sections']:
            print(f"   - {section['heading']} ({section['words']}w): {section['content'][:80]}...")
        
        print()
        print()
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze blind spots to show private content and expected public output'
    )
    parser.add_argument('two_pole_output', help='Path to two_pole output directory')
    parser.add_argument('data_room', help='Path to data room directory')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top blind spots to analyze')
    
    args = parser.parse_args()
    
    analyze_blind_spots(args.two_pole_output, args.data_room, args.top_n)
