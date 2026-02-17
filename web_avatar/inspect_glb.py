"""Quick GLB inspector - check bone positions, mesh bounds, and transforms."""
import struct
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "models/arknights_rigged.glb"

with open(path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    total_len = struct.unpack('<I', f.read(4))[0]
    
    # JSON chunk
    chunk_len = struct.unpack('<I', f.read(4))[0]
    chunk_type = f.read(4)
    gltf = json.loads(f.read(chunk_len).decode('utf-8'))
    
    # Binary chunk
    bin_chunk_len = struct.unpack('<I', f.read(4))[0]
    bin_chunk_type = f.read(4)
    bin_data = f.read(bin_chunk_len)

print(f"=== GLB: {path} ===")
print(f"Version: {version}, Size: {total_len} bytes")

# Root nodes
print("\n=== ROOT NODES ===")
scene_nodes = gltf['scenes'][0]['nodes']
for ni in scene_nodes:
    node = gltf['nodes'][ni]
    name = node.get('name', f'node_{ni}')
    rot = node.get('rotation', 'none')
    trans = node.get('translation', 'none')
    scale = node.get('scale', 'none')
    children = node.get('children', [])
    has_mesh = 'mesh' in node
    has_skin = 'skin' in node
    print(f"  Node[{ni}] {name}: rot={rot}, pos={trans}, mesh={has_mesh}, skin={has_skin}, children={len(children)}")
    
    # Show children
    for ci in children:
        cn = gltf['nodes'][ci]
        cname = cn.get('name', f'node_{ci}')
        crot = cn.get('rotation', 'none')
        ctrans = cn.get('translation', 'none')
        cmesh = 'mesh' in cn
        cskin = 'skin' in cn
        print(f"    Child[{ci}] {cname}: rot={crot}, pos={ctrans}, mesh={cmesh}, skin={cskin}")

# Find mesh vertex positions
print("\n=== MESH VERTEX BOUNDS ===")
for mi, mesh in enumerate(gltf.get('meshes', [])):
    print(f"Mesh[{mi}] '{mesh.get('name', '?')}':")
    for pi, prim in enumerate(mesh.get('primitives', [])):
        pos_acc_idx = prim.get('attributes', {}).get('POSITION')
        if pos_acc_idx is not None:
            accessor = gltf['accessors'][pos_acc_idx]
            bv = gltf['bufferViews'][accessor['bufferView']]
            offset = bv.get('byteOffset', 0) + accessor.get('byteOffset', 0)
            count = accessor['count']
            stride = bv.get('byteStride', 12)
            
            # Read positions
            min_pos = accessor.get('min', [0, 0, 0])
            max_pos = accessor.get('max', [0, 0, 0])
            print(f"  Primitive[{pi}]: {count} vertices")
            print(f"    min: X={min_pos[0]:.4f}, Y={min_pos[1]:.4f}, Z={min_pos[2]:.4f}")
            print(f"    max: X={max_pos[0]:.4f}, Y={max_pos[1]:.4f}, Z={max_pos[2]:.4f}")
            
            # Check nodeTransform that might affect this mesh
            for ni2, node in enumerate(gltf['nodes']):
                if node.get('mesh') == mi:
                    ntrans = node.get('translation', 'none')
                    nrot = node.get('rotation', 'none')
                    nscale = node.get('scale', 'none')
                    print(f"    Node[{ni2}]: trans={ntrans}, rot={nrot}, scale={nscale}")

# Check skin inverse bind matrices
if 'skins' in gltf:
    skin = gltf['skins'][0]
    ibm_acc = skin.get('inverseBindMatrices')
    if ibm_acc is not None:
        acc = gltf['accessors'][ibm_acc]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        
        print(f"\n=== INVERSE BIND MATRICES ({count}) ===")
        # Read first few matrices (4x4 float32 = 64 bytes each)
        joints = skin['joints']
        for i in range(min(5, count)):
            mat_offset = offset + i * 64
            mat = struct.unpack_from('<16f', bin_data, mat_offset)
            name = gltf['nodes'][joints[i]].get('name', '?')
            # Column-major: translation is at indices 12, 13, 14
            tx, ty, tz = mat[12], mat[13], mat[14]
            print(f"  {name}: IBMtranslation=({tx:.4f}, {ty:.4f}, {tz:.4f})")

# Orientation check
print("\n=== ORIENTATION SUMMARY ===")
for mi, mesh in enumerate(gltf.get('meshes', [])):
    for pi, prim in enumerate(mesh.get('primitives', [])):
        pos_acc_idx = prim.get('attributes', {}).get('POSITION')
        if pos_acc_idx is not None:
            accessor = gltf['accessors'][pos_acc_idx]
            min_p = accessor.get('min', [0, 0, 0])
            max_p = accessor.get('max', [0, 0, 0])
            range_x = max_p[0] - min_p[0]
            range_y = max_p[1] - min_p[1]
            range_z = max_p[2] - min_p[2]
            print(f"Mesh range: X={range_x:.4f} (width), Y={range_y:.4f} (should be height), Z={range_z:.4f} (depth)")
            if range_y > range_x and range_y > range_z:
                print("  -> Y is tallest axis (CORRECT for Y-up)")
            elif range_z > range_y:
                print("  -> Z is tallest axis (WRONG - model may be sideways or Z-up!)")
            elif range_x > range_y:
                print("  -> X is tallest axis (WRONG - model is lying down?!)")

