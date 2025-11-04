#!/usr/bin/env python3
"""
Import Anthropic Claude Data Export to Mathlete Data Room Format

Processes Anthropic Claude data export ZIP file and converts conversations
to the standardized AI archive format for two-pole analysis.

Usage:
    python import_anthropic_archive.py <anthropic_export.zip> <data_room_path>
"""
import json
import zipfile
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_anthropic_conversation(conv_data):
    """
    Parse a single Anthropic Claude conversation from the conversations.json format.
    
    Args:
        conv_data: Single conversation object from Anthropic export
        
    Returns:
        dict with parsed conversation data
    """
    # Extract metadata
    title = conv_data.get('name', 'Untitled Conversation')
    created_at = conv_data.get('created_at')
    updated_at = conv_data.get('updated_at')
    conv_uuid = conv_data.get('uuid')
    
    # Convert timestamps to dates
    if created_at:
        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            conv_date = date_obj.strftime('%Y-%m-%d')
        except:
            conv_date = datetime.now().strftime('%Y-%m-%d')
    else:
        conv_date = datetime.now().strftime('%Y-%m-%d')
    
    # Extract chat messages
    chat_messages = conv_data.get('chat_messages', [])
    if not chat_messages:
        return None
    
    # Build message arrays
    text_parts = []
    parsed_messages = []
    
    for msg in chat_messages:
        sender = msg.get('sender', 'unknown')
        
        # Skip if not human or assistant
        if sender not in ['human', 'assistant']:
            continue
        
        # Extract text from content array
        content = msg.get('content', [])
        text = msg.get('text', '')  # Use text field if available
        
        # If no text field, try to extract from content
        if not text:
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                    text = content_item.get('text', '')
                    break
        
        if not text or text.strip() == '':
            continue
        
        # Format for text field
        role_label = "Human" if sender == "human" else "Assistant"
        text_parts.append(f"{role_label}: {text}")
        
        # Parse for messages array
        created_timestamp = msg.get('created_at')
        if created_timestamp:
            try:
                timestamp_str = created_timestamp
                # Ensure proper format
                if not timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str.replace('Z', '') + 'Z'
            except:
                timestamp_str = None
        else:
            timestamp_str = None
        
        parsed_messages.append({
            'role': sender,
            'content': text,
            'timestamp': timestamp_str
        })
    
    # Skip if still no content
    if not text_parts or not parsed_messages:
        return None
    
    # Combine text
    full_text = "\n\n".join(text_parts)
    
    # Create unique ID from UUID (first 8 chars)
    conv_id = conv_uuid[:8] if conv_uuid else 'unknown'
    
    result = {
        'id': f"claude_{conv_id}",
        'title': title,
        'text': full_text,
        'date': conv_date,
        'source': 'anthropic',
        'messages': parsed_messages,
        'metadata': {
            'uuid': conv_uuid,
            'created_at': created_at,
            'updated_at': updated_at,
            'conversation_length': len(parsed_messages),
            'human_contributions': sum(1 for m in parsed_messages if m['role'] == 'human'),
            'ai_contributions': sum(1 for m in parsed_messages if m['role'] == 'assistant')
        }
    }
    
    return result


def process_anthropic_archive(archive_path, data_room_path):
    """
    Process Anthropic Claude export ZIP file and extract conversations.
    
    Args:
        archive_path: Path to Anthropic export ZIP file
        data_room_path: Path to mathlete data room directory
    """
    archive_path = Path(archive_path)
    data_room_path = Path(data_room_path)
    
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        sys.exit(1)
    
    print(f"📥 Processing Anthropic Claude archive: {archive_path.name}")
    print("=" * 80)
    
    # Create ai_archives/anthropic directory structure
    ai_archives_dir = data_room_path / "ai_archives" / "anthropic"
    ai_archives_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {ai_archives_dir}")
    
    # Extract conversations.json from ZIP
    conversations_data = None
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Look for conversations.json
            if 'conversations.json' in zip_ref.namelist():
                conversations_data = json.loads(zip_ref.read('conversations.json'))
                print(f"✓ Loaded conversations.json from archive")
            else:
                print(f"❌ No conversations.json found in archive")
                print(f"   Available files: {zip_ref.namelist()}")
                sys.exit(1)
    except zipfile.BadZipFile:
        print(f"❌ Invalid ZIP file: {archive_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse conversations.json: {e}")
        sys.exit(1)
    
    # Process each conversation
    print(f"\n📝 Processing {len(conversations_data)} conversations...")
    
    stats = defaultdict(int)
    success_count = 0
    skip_count = 0
    
    for idx, conv_data in enumerate(conversations_data):
        try:
            parsed = parse_anthropic_conversation(conv_data)
            
            if not parsed:
                skip_count += 1
                continue
            
            # Determine output directory based on date
            date = parsed['date']
            year, month, day = date.split('-')
            output_dir = ai_archives_dir / year / month / day
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename from ID and title
            title_slug = parsed['title'].replace(' ', '_').replace('/', '_').replace('\\', '_')
            title_slug = ''.join(c for c in title_slug if c.isalnum() or c in ('_', '-'))[:50]
            filename = f"{parsed['id']}-{title_slug}.json"
            output_path = output_dir / filename
            
            # Save conversation
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            
            success_count += 1
            stats[date] += 1
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(conversations_data)}...")
        
        except Exception as e:
            print(f"  ⚠️  Error processing conversation {idx}: {e}")
            skip_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ Import Complete!")
    print("=" * 80)
    
    print(f"\n📊 Statistics:")
    print(f"  Total conversations: {len(conversations_data)}")
    print(f"  Successfully imported: {success_count}")
    print(f"  Skipped (empty/error): {skip_count}")
    
    if stats:
        print(f"\n  By date:")
        for date in sorted(stats.keys())[:20]:  # Show first 20 dates
            print(f"    {date}: {stats[date]} conversations")
        if len(stats) > 20:
            print(f"    ... and {len(stats) - 20} more dates")
    
    print(f"\n💾 Output location: {ai_archives_dir}")
    print(f"\n📁 Next steps:")
    print(f"   1. Verify imports with: ls -R {ai_archives_dir}")
    print(f"   2. Run two-pole analysis:")
    print(f"      python scripts/run_two_pole_pipeline.py {data_room_path}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Import Anthropic Claude export to mathlete data room format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import Anthropic export to data room
  python import_anthropic_archive.py claude-export.zip ~/mathlete-data-room

  # Import from current directory
  python import_anthropic_archive.py data-*.zip ~/mathlete-data-room

Output Structure:
  <data_room>/ai_archives/anthropic/YYYY/MM/DD/*.json

See DATA_ROOM_101_AI_ARCHIVES.md for format specification.
        """
    )
    
    parser.add_argument('archive', type=str,
                       help='Path to Anthropic export ZIP file')
    parser.add_argument('data_room', type=str,
                       help='Path to mathlete data room directory')
    
    args = parser.parse_args()
    
    try:
        success_count = process_anthropic_archive(args.archive, args.data_room)
        if success_count == 0:
            print("\n⚠️  No conversations were imported. Check the archive format.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Import interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

