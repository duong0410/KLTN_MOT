#!/usr/bin/env python3
"""
Script to check BoxMOT structure and available classes
"""

import sys

print("Checking BoxMOT installation and structure...")
print("=" * 80)

try:
    import boxmot
    print(f"✓ BoxMOT imported successfully")
    print(f"  Version: {boxmot.__version__ if hasattr(boxmot, '__version__') else 'Unknown'}")
    print(f"  Location: {boxmot.__file__}")
    print()
    
    # Check what's available in boxmot
    print("Available in boxmot module:")
    print("-" * 80)
    items = dir(boxmot)
    for item in items:
        if not item.startswith('_'):
            print(f"  - {item}")
    print()
    
    # Try to import trackers
    print("Checking boxmot.trackers:")
    print("-" * 80)
    try:
        from boxmot import trackers
        print(f"✓ boxmot.trackers imported")
        print(f"  Available: {dir(trackers)}")
        print()
    except ImportError as e:
        print(f"✗ Cannot import trackers: {e}")
        print()
    
    # Try specific tracker imports
    print("Trying specific tracker imports:")
    print("-" * 80)
    
    tracker_attempts = [
        ("BYTETracker", "from boxmot import BYTETracker"),
        ("ByteTrack", "from boxmot import ByteTrack"),
        ("ByTracker", "from boxmot import ByTracker"),
        ("bytetrack", "from boxmot.trackers import bytetrack"),
        ("BYTETracker from trackers", "from boxmot.trackers.bytetrack import BYTETracker"),
        ("BYTETracker from byte_tracker", "from boxmot.trackers.bytetrack.byte_tracker import BYTETracker"),
        ("ByteTrack from trackers", "from boxmot.trackers.bytetrack import ByteTrack"),
    ]
    
    successful_imports = []
    
    for name, import_str in tracker_attempts:
        try:
            exec(import_str)
            print(f"✓ {name}: {import_str}")
            successful_imports.append((name, import_str))
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    print()
    print("=" * 80)
    if successful_imports:
        print("SUCCESSFUL IMPORTS:")
        for name, import_str in successful_imports:
            print(f"  {import_str}")
    else:
        print("No successful imports found.")
        print("\nTrying to explore boxmot.trackers structure...")
        
        try:
            import os
            import boxmot
            boxmot_path = os.path.dirname(boxmot.__file__)
            trackers_path = os.path.join(boxmot_path, 'trackers')
            
            if os.path.exists(trackers_path):
                print(f"\nTrackers directory: {trackers_path}")
                print("Contents:")
                for item in os.listdir(trackers_path):
                    print(f"  - {item}")
                    
                    # Check bytetrack directory
                    if item == 'bytetrack' or item.startswith('byte'):
                        subpath = os.path.join(trackers_path, item)
                        if os.path.isdir(subpath):
                            print(f"    Contents of {item}:")
                            for subitem in os.listdir(subpath):
                                print(f"      - {subitem}")
        except Exception as e:
            print(f"Error exploring structure: {e}")
    
except ImportError as e:
    print(f"✗ BoxMOT not installed: {e}")
    print("\nPlease install BoxMOT:")
    print("  pip install boxmot")
    sys.exit(1)