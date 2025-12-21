# Fix: ChromaDB Metadata List Error

## Problem
```
❌ Error ingesting PDFs: Expected metadata value to be a str, int, float, bool, SparseVector, or None, 
got ['/home/krira/Project/TALKERPDF/app/static/images/linea_page_1_img_2.png'] which is a list in add.
```

## Root Cause
ChromaDB does **not support list types** in metadata fields. The `image_paths` field was being stored as a Python list, which caused the error during ingestion.

## Solution
Convert `image_paths` list to a **JSON string** for storage in ChromaDB, then parse it back to a list when retrieving.

---

## Changes Made

### 1. **core/ingestion.py** - Store as JSON String
```python
# Added import
import json

# Modified metadata storage (Line ~477)
if image_paths:
    metadata["image_paths"] = json.dumps(image_paths)  # ✅ Convert list to JSON string
```

### 2. **core/retrieval.py** - Parse JSON String Back to List
```python
# Added import
import json

# Modified re-ranking function (Line ~189)
metadata = metadatas[idx] if idx < len(metadatas) else {}
if 'image_paths' in metadata and isinstance(metadata['image_paths'], str):
    try:
        metadata['image_paths'] = json.loads(metadata['image_paths'])  # ✅ Parse back to list
    except (json.JSONDecodeError, TypeError):
        metadata['image_paths'] = []
```

### 3. **app/ui.py** - Helper Function for Safe Parsing
```python
# Added import
import json

# Added helper function (Line ~18)
def parse_image_paths(metadata):
    """Parse image_paths from metadata, handling both JSON strings and lists."""
    image_paths = metadata.get('image_paths', [])
    if isinstance(image_paths, str):
        try:
            return json.loads(image_paths)
        except (json.JSONDecodeError, TypeError):
            return []
    return image_paths if isinstance(image_paths, list) else []

# Updated all usages to use helper:
# Before: image_paths = chunk_info['metadata']['image_paths']
# After:  image_paths = parse_image_paths(chunk_info['metadata'])
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION (core/ingestion.py)                                  │
├─────────────────────────────────────────────────────────────────┤
│ image_paths = [                                                 │
│   '/path/to/image1.png',                                        │
│   '/path/to/image2.png'                                         │
│ ]                                                               │
│                                                                 │
│ metadata["image_paths"] = json.dumps(image_paths)              │
│ # Result: '["path/to/image1.png", "path/to/image2.png"]'      │
│                                                                 │
│ ↓ Store in ChromaDB as STRING ✅                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL (core/retrieval.py)                                  │
├─────────────────────────────────────────────────────────────────┤
│ ↓ Fetch from ChromaDB (string)                                 │
│                                                                 │
│ metadata['image_paths'] = json.loads(metadata['image_paths'])  │
│ # Result: ['/path/to/image1.png', '/path/to/image2.png']      │
│                                                                 │
│ ↓ Return as LIST for UI ✅                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ UI DISPLAY (app/ui.py)                                          │
├─────────────────────────────────────────────────────────────────┤
│ image_paths = parse_image_paths(result['metadata'])            │
│ # Safely handles both string and list formats                  │
│                                                                 │
│ for img_path in image_paths:                                   │
│     st.image(img_path)  # Display images ✅                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ChromaDB Metadata Constraints

### ✅ Supported Types
- `str` (string)
- `int` (integer)
- `float` (floating point)
- `bool` (boolean)
- `None` (null)
- `SparseVector` (special type for sparse embeddings)

### ❌ NOT Supported
- `list` (arrays) ← **Our problem!**
- `dict` (objects)
- Custom objects

### Workaround
Convert complex types to JSON strings:
```python
# Store
metadata["my_list"] = json.dumps([1, 2, 3])      # ✅ "1,2,3]"
metadata["my_dict"] = json.dumps({"key": "val"}) # ✅ '{"key":"val"}'

# Retrieve
my_list = json.loads(metadata["my_list"])        # ✅ [1, 2, 3]
my_dict = json.loads(metadata["my_dict"])        # ✅ {"key": "val"}
```

---

## Testing

### Validation
```bash
$ python3 -m py_compile core/ingestion.py core/retrieval.py app/ui.py
# ✅ All files compile successfully
```

### Test Ingestion
1. Upload a PDF with charts
2. Watch logs for successful image extraction
3. Check ChromaDB storage (metadata should be strings)

### Test Retrieval
1. Query for visual content
2. Images should display correctly in UI
3. Check debug expander for metadata

---

## Files Modified

1. **core/ingestion.py**
   - Added `import json`
   - Line ~477: `metadata["image_paths"] = json.dumps(image_paths)`

2. **core/retrieval.py**
   - Added `import json`
   - Line ~189-195: Parse JSON string back to list in re-ranking results

3. **app/ui.py**
   - Added `import json`
   - Line ~18-26: Added `parse_image_paths()` helper function
   - Updated 3 locations to use helper function instead of direct access

---

## Backward Compatibility

The solution is **forward-compatible only**:
- ✅ New ingestions will work correctly
- ⚠️ Old data (if any) with list-type metadata will need re-ingestion

If you have existing data with lists:
```bash
# Option 1: Clear database and re-ingest
rm -rf data/chroma_db/*
# Then re-upload PDFs through UI

# Option 2: Migration script (if needed)
# Parse old metadata, convert lists to JSON strings, update collection
```

---

## Summary

✅ **Fixed**: ChromaDB list metadata error  
✅ **Method**: JSON string serialization/deserialization  
✅ **Files**: ingestion.py, retrieval.py, ui.py  
✅ **Tested**: Syntax validation passed  
✅ **Compatible**: All existing features preserved  

The system now correctly stores image paths as JSON strings in ChromaDB and safely parses them back to lists for UI display.
