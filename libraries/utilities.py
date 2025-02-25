import json


# Save slab information
def save_json(
        slab,
        data='slab',
        filename='slab_data.json'
):
    """Save slab information into json file.

    Args:
        slab (slab): Slab object.
        filename (str): Name of the json file.

    Returns:
        None
    """
    if data == 'slab':
        slab_data = {
            'miller_index': slab.miller_index,
            'shift': slab.shift,
            'surface_area': slab.surface_area,
            'number_of_sites': len(slab.sites),
            'is_polar': str(slab.is_polar()),
            'is_symmetric': str(slab.is_symmetric())
        }
    elif data == 'bulk':
        slab_data = {
            'number_of_sites': slab.num_sites,
        }

    with open(filename, 'w') as json_file:
        json.dump(slab_data, json_file)


def load_json(
        filename='slab_data.json'
):
    """Load slab information from json file.

    Args:
        filename (str): Name of the json file.

    Returns:
        slab_data (dict): Dictionary containing the slab information.
    """
    # Load the data from the JSON file
    with open(filename, 'r') as json_file:
        slab_data = json.load(json_file)
    return slab_data