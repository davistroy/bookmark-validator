#!/usr/bin/env python3
"""
Regenerate CSV with fixed tags from checkpoint data.
"""

import json
import csv
from pathlib import Path

# Load checkpoint
checkpoint_file = Path('.bookmark_checkpoints/checkpoint_20260131_213155.json')
print(f"Loading checkpoint: {checkpoint_file}")

with open(checkpoint_file, 'r') as f:
    checkpoint = json.load(f)

# Extract data
validated_bookmarks = checkpoint.get('validated_bookmarks', [])
tag_assignments = checkpoint.get('tag_assignments', {})
ai_results = checkpoint.get('ai_results', {})

print(f"Found {len(validated_bookmarks)} validated bookmarks")
print(f"Found {len(tag_assignments)} tag assignments")
print(f"Found {len(ai_results)} AI results")

# Build output rows
output_rows = []

for item in validated_bookmarks:
    bookmark = item.get('bookmark', {})
    validation = item.get('validation', {})

    # Only include valid bookmarks
    if not validation.get('is_valid', False):
        continue

    url = bookmark.get('url', '')

    # Get tags (should be a list)
    tags = tag_assignments.get(url, [])

    # Format tags properly for Raindrop.io
    if not tags:
        formatted_tags = ""
    elif len(tags) == 1:
        formatted_tags = tags[0]
    else:
        formatted_tags = f'"{", ".join(tags)}"'

    # Get AI description
    ai_result = ai_results.get(url)
    note = ai_result.get('enhanced_description', '') if ai_result else bookmark.get('note', '')

    # Get created date
    created = bookmark.get('created', '')

    # Build row
    row = {
        'url': url,
        'folder': bookmark.get('folder', ''),
        'title': bookmark.get('title', ''),
        'note': note,
        'tags': formatted_tags,
        'created': created,
    }

    output_rows.append(row)

print(f"Prepared {len(output_rows)} valid bookmarks for export")

# Write to CSV
output_file = 'enhanced_bookmarks_corrected.csv'
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['url', 'folder', 'title', 'note', 'tags', 'created'])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\n[OK] Corrected CSV saved to: {output_file}")
print(f"[OK] Total bookmarks: {len(output_rows)}")

# Show sample
print("\nSample of corrected tags:")
print("=" * 80)
for i, row in enumerate(output_rows[:5]):
    print(f"{i+1}. {row['title'][:60]}")
    print(f"   Tags: {row['tags']}")
    print()
