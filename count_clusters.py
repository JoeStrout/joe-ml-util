"""
Given a path to a precomputed volume containing a segmentation layer, this reports
on the number and range of segment IDs therein.
"""

from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.geometry import Vec3D, BBox3D
import cc3d
import numpy as np
import zetta_utils.tensor_ops.convert as convert
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec
from caveclient import CAVEclient
import nglui

def verify_cave_auth():
    global client
    client = CAVEclient()
    try:
        client.state
        return  # no exception?  All's good!
    except:
        pass
    print('Authentication needed.')
    print('Go to: https://global.daf-apis.com/auth/api/v1/create_token')
    token = input('Enter token: ')
    client.auth.save_token(token=token)

def inspect_layer(path):
    """Load a CloudVolume and report its contents based on just the path."""
    spec = PrecomputedInfoSpec(reference_path=path)
    info = spec.make_info()
    print(f'{path}')
    print(f'{info["data_type"]} {info["type"]} layer')
    print(f'{len(info["scales"])} scale(s):')
    for scale in info["scales"]:
        start_coord = scale['voxel_offset']
        size_in_voxels = scale['size']
        end_coord = [a+b for (a,b) in zip(start_coord, size_in_voxels)]
        res = scale["resolution"]
        print(f' - res: {res}, bounds {start_coord} - {end_coord}')
        size_in_nm = [v*r for (v,r) in zip(size_in_voxels, res)]
        size_in_µm = [s*0.001 for s in size_in_nm]
        vol_in_voxels = size_in_voxels[0] * size_in_voxels[1] * size_in_voxels[2]
        vol_in_µm = size_in_µm[0] * size_in_µm[1] * size_in_µm[2]
        print(f'   size: {size_in_voxels} voxels ({vol_in_voxels} vox^3); {size_in_µm} µm ({vol_in_µm} µm^3)')

def load_volume(path, scale_index=0):
    """
    Load a CloudVolume given the path, and optionally, which scale (resolution) is desired.
    Return the CloudVolume, and a BBox3D describing the data bounds.
    """
    spec = PrecomputedInfoSpec(reference_path=path)
    info = spec.make_info()
    scale = info["scales"][scale_index]
    resolution = scale["resolution"]
    start_coord = scale['voxel_offset']
    size = scale['size']
    end_coord = [a+b for (a,b) in zip(start_coord, size)]
    cvl = build_cv_layer(path=path,
                         allow_slice_rounding=True,
                         default_desired_resolution=resolution,
                         index_resolution=resolution,
                         data_resolution=resolution,
                         interpolation_mode=info["type"])
    bounds = BBox3D.from_coords(start_coord, end_coord)
    return cvl, bounds

def count_clusters(seg_path):
    # load cloud volume
    inspect_layer(seg_path)
    cvl, bounds = load_volume(seg_path)
    m = cvl[bounds.start.x:bounds.end.x, bounds.start.y:bounds.end.y, bounds.start.z:bounds.end.z]
    ids = np.unique(m[m != 0])
    print(f'{len(ids)} clusters, ranging from {ids[0]} to {ids[-1]}')

def count_all_from_NG_state(ng_path):
    verify_cave_auth()
    state_id = ng_path.split('/')[-1] # get state ID from the end of the NG state URL
    global state  # HACK
    state = client.state.get_state_json(state_id)
    for layer in state['layers']:
        if layer['type'] == 'segmentation':
            count_clusters(layer['source'])

if __name__ == "__main__":
    print('Enter path to CV layer or NG state:')
    # Examples:
    #  precomputed://gs://dkronauer-ant-001-manual-labels/synapses/664bbece010000c600388f09/postsynaptic-terminal/000
    #  https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6666368729481216
    path = input('>')

    if 'nglstate' in path:
        count_all_from_NG_state(path)
    else:
        count_clusters(path)

