#!/usr/bin/env python3
"""
Import OpenAI ChatGPT Data Export to Mathlete Data Room Format

Processes OpenAI/ChatGPT data export ZIP file and converts conversations
to the standardized AI archive format for two-pole analysis.

Usage:
    python import_openai_archive.py <openai_export.zip> <data_room_path>
"""
import json
import zipfile
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_openai_conversation(conv_data):
    """
    Parse a single OpenAI conversation from the conversations.json format.
    
    Args:
        conv_data: Single conversation object from OpenAI export
        
    Returns:
        dict with parsed conversation data
    """
    # Extract metadata
    title = conv_data.get('title', 'Untitled Conversation')
    create_time = conv_data.get('create_time')
    update_time = conv_data.get('update_time')
    
    # Convert timestamps to dates
    if create_time:
        conv_date = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d')
    else:
        conv_date = datetime.now().strftime('%Y-%m-%d')
    
    # Build message chain from mapping
    mapping = conv_data.get('mapping', {})
    if not mapping:
        return None
    
    # Find root message
    root_ids = []
    messages_by_id = {}
    
    for msg_id, msg_data in mapping.items():
        messages_by_id[msg_id] = msg_data
        parent = msg_data.get('parent')
        if parent is None or parent == 'client-created-root':
            root_ids.append(msg_id)
    
    # Reconstruct message tree (topological sort)
    def collect_messages(start_id, collected):
        """Recursively collect messages in order."""
        if start_id not in messages_by_id:
            return
        
        msg_data = messages_by_id[start_id]
        msg_obj = msg_data.get('message')
        
        # Only include user and assistant messages (skip system)
        if msg_obj and msg_obj.get('author', {}).get('role') in ['user', 'assistant']:
            collected.append(msg_obj)
        
        # Recurse to children
        for child_id in msg_data.get('children', []):
            collect_messages(child_id, collected)
    
    messages = []
    for root_id in root_ids:
        collect_messages(root_id, messages)
    
    # Skip if no valid messages
    if not messages:
        return None
    
    # Build conversation text
    text_parts = []
    parsed_messages = []
    
    for msg in messages:
        role = msg.get('author', {}).get('role', 'unknown')
        
        # Extract content
        content_obj = msg.get('content', {})
        if content_obj.get('content_type') == 'text':
            parts = content_obj.get('parts', [])
            content = '\n'.join([p for p in parts if isinstance(p, str)])
        else:
            # Handle other content types (e.g., multimodal)
            content = f"[{content_obj.get('content_type', 'unknown')} content]"
        
        if not content or content.strip() == '':
            continue
        
        # Format for text field
        role_label = "Human" if role == "user" else "Assistant"
        text_parts.append(f"{role_label}: {content}")
        
        # Parse for messages array
        create_timestamp = msg.get('create_time')
        if create_timestamp:
            try:
                timestamp_str = datetime.fromtimestamp(create_timestamp).isoformat() + 'Z'
            except:
                timestamp_str = None
        else:
            timestamp_str = None
        
        parsed_messages.append({
            'role': role,
            'content': content,
            'timestamp': timestamp_str
        })
    
    # Skip if still no content
    if not text_parts:
        return None
    
    # Combine text
    full_text = "\n\n".join(text_parts)
    
    # Create unique ID from first 8 chars of original ID
    conv_id = list(mapping.keys())[0][:8] if mapping else 'unknown'
    
    result = {
        'id': f"openai_{conv_id}",
        'title': title,
        'text': full_text,
        'date': conv_date,
        'source': 'openai',
        'messages': parsed_messages,
        'metadata': {
            'create_time': create_time,
            'update_time': update_time,
            'conversation_length': len(parsed_messages),
            'human_contributions': sum(1 for m in parsed_messages if m['role'] == 'user'),
            'ai_contributions': sum(1 for m in parsed_messages if m['role'] == 'assistant')
        }
    }
    
    return result


def process_openai_archive(archive_path, data_room_path):
    """
    Process OpenAI export ZIP file and extract conversations.
    
    Args:
        archive_path: Path to OpenAI export ZIP file
        data_room_path: Path to mathlete data room directory
    """
    archive_path = Path(archive_path)
    data_room_path = Path(data_room_path)
    
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        sys.exit(1)
    
    print(f"📥 Processing OpenAI archive: {archive_path.name}")
    print("=" * 80)
    
    # Create ai_archives/openai directory structure
    ai_archives_dir = data_room_path / "ai_archives" / "openai"
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
            parsed = parse_openai_conversation(conv_data)
            
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
            if (idx + 1) % 50 == 0:
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
        for date in sorted(stats.keys()):
            print(f"    {date}: {stats[date]} conversations")
    
    print(f"\n💾 Output location: {ai_archives_dir}")
    print(f"\n📁 Next steps:")
    print(f"   1. Verify imports with: ls -R {ai_archives_dir}")
    print(f"   2. Run two-pole analysis:")
    print(f"      python scripts/run_two_pole_pipeline.py {data_room_path}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Import OpenAI ChatGPT export to mathlete data room format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import OpenAI export to data room
  python import_openai_archive.py openai-export.zip ~/mathlete-data-room

  # Import from current directory
  python import_openai_archive.py conversations-*.zip ~/mathlete-data-room

Output Structure:
  <data_room>/ai_archives/openai/YYYY/MM/DD/*.json

See DATA_ROOM_101_AI_ARCHIVES.md for format specification.
        """
    )
    
    parser.add_argument('archive', type=str,
                       help='Path to OpenAI export ZIP file')
    parser.add_argument('data_room', type=str,
                       help='Path to mathlete data room directory')
    
    args = parser.parse_args()
    
    try:
        success_count = process_openai_archive(args.archive, args.data_room)
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

