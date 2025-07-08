import json
import numpy as np 

def prepare_metadata_for_weaviate(metadata: dict) -> dict:
    """
    Transforms a document's metadata dictionary to be compatible with Weaviate.
    Specifically handles the 'coordinates' field by converting numpy.float64 to
    standard floats and flattening the 'points' array into 'points_flat'.

    Args:
        metadata (dict): The original metadata dictionary from an UnstructuredLoader document.

    Returns:
        dict: A new metadata dictionary with data types compatible with Weaviate.
    """
    weaviate_compatible_metadata = metadata.copy() 

    # --- Handle 'coordinates' field ---
    if 'coordinates' in weaviate_compatible_metadata and weaviate_compatible_metadata['coordinates'] is not None:
        coords = weaviate_compatible_metadata['coordinates']

        if not isinstance(coords, dict):
            weaviate_compatible_metadata.pop('coordinates', None)
        else:
            original_points = coords.get('points')

            bbox_points_list = [] # This will store our list of point objects

            if original_points and len(original_points) == 4:
                try:
                    # Iterate through each (x, y) tuple and assign to named properties
                    for i, (x_coord, y_coord) in enumerate(original_points):
                        point_obj = {}
                        point_obj[f'x{i}'] = float(x_coord) # Use f-string for x0, x1, etc.
                        point_obj[f'y{i}'] = float(y_coord) # Use f-string for y0, y1, etc.
                        bbox_points_list.append(point_obj)

                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error processing 'points' for bbox_points list: {e}. Original points: {original_points}")
                    bbox_points_list = [] # Reset to empty on error
            else:
                print(f"Warning: 'points' missing or not 4 elements for bbox_points list. Points: {original_points}")

            if bbox_points_list:
                weaviate_compatible_metadata['bbox_points'] = bbox_points_list

            # Remove the original 'coordinates' key
            weaviate_compatible_metadata.pop('coordinates', None)
    # --- Handle other common metadata fields, ensuring compatible types ---
    # Convert 'last_modified' to string if it's a datetime object or similar
    if 'last_modified' in weaviate_compatible_metadata and weaviate_compatible_metadata['last_modified'] is not None:
        if not isinstance(weaviate_compatible_metadata['last_modified'], str):
            try:
                # Attempt to convert to ISO format if it's a datetime object
                if hasattr(weaviate_compatible_metadata['last_modified'], 'isoformat'):
                    weaviate_compatible_metadata['last_modified'] = weaviate_compatible_metadata['last_modified'].isoformat()
                else:
                    weaviate_compatible_metadata['last_modified'] = str(weaviate_compatible_metadata['last_modified'])
            except Exception as e:
                print(f"Warning: Could not convert 'last_modified' to string: {e}. Original: {weaviate_compatible_metadata['last_modified']}")
                weaviate_compatible_metadata['last_modified'] = None # Or keep original if conversion fails
    
    # Ensure 'page_number' is an integer
    if 'page_number' in weaviate_compatible_metadata and weaviate_compatible_metadata['page_number'] is not None:
        try:
            weaviate_compatible_metadata['page_number'] = int(weaviate_compatible_metadata['page_number'])
        except (TypeError, ValueError):
            print(f"Warning: Could not convert 'page_number' '{weaviate_compatible_metadata['page_number']}' to int.")
            weaviate_compatible_metadata['page_number'] = 0 # Default or remove
    
    # Ensure 'languages' is a list of strings
    if 'languages' in weaviate_compatible_metadata and weaviate_compatible_metadata['languages'] is not None:
        if not isinstance(weaviate_compatible_metadata['languages'], list):
            weaviate_compatible_metadata['languages'] = [str(weaviate_compatible_metadata['languages'])] # Wrap in list
        else:
            weaviate_compatible_metadata['languages'] = [str(lang) for lang in weaviate_compatible_metadata['languages']] # Ensure elements are strings

    # Ensure other text fields are strings
    for key in ['source', 'filetype', 'filename', 'category', 'element_id']:
        if key in weaviate_compatible_metadata and weaviate_compatible_metadata[key] is not None:
            weaviate_compatible_metadata[key] = str(weaviate_compatible_metadata[key])

    return weaviate_compatible_metadata

if __name__ == "__main__":

    import numpy as np

    sample_metadata_1 = {
        'source': 'ncert_crop.pdf',
        'coordinates': {
            'points': (
                (np.float64(25.273810555555556), np.float64(-63.814965699999796)),
                (np.float64(25.273810555555556), np.float64(2340.0843641666665)),
                (np.float64(1724.7237358888888), np.float64(2340.0843641666665)),
                (np.float64(1724.7237358888888), np.float64(-63.814965699999796))
            ),
            'system': 'PixelSpace',
            'layout_width': 1750,
            'layout_height': 2280
        },
        'last_modified': '2025-07-08T10:50:56',
        'filetype': 'application/pdf',
        'languages': ['eng'],
        'page_number': 1,
        'filename': 'ncert_crop.pdf',
        'category': 'Image',
        'element_id': '4b9f10d95e3a64071056d4d5ba6c2eb2'
    }

    print("--- Original Metadata 1 ---")
    print(json.dumps(sample_metadata_1, indent=2, default=str)) # default=str handles np.float64 for printing
    processed_metadata_1 = prepare_metadata_for_weaviate(sample_metadata_1)
    print("\n--- Processed Metadata 1 (Weaviate Compatible) ---")
    print(json.dumps(processed_metadata_1, indent=2))
    print(f"Type of bbox elements: {type(processed_metadata_1['bbox']['x0'])}")