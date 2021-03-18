import model
import bspt
import utils

import os
import torch
import torchvision
import numpy as np
import PIL


IMAGE_TRANSFORM = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale ( num_output_channels=1 ),
            torchvision.transforms.Resize    ( size=(128,128) ),
            #torchvision.transforms.Grayscale ( num_output_channels=1 ),
            torchvision.transforms.ToTensor  (),                        # This will also normalize the image in the range [0.0, 1.0].
        ])


def reconstruct_object(image_path, checkpoint_path, output_path):
    device = torch.device('cpu:0')

    real_size             = 64 # Output point-value voxel grid size in testing.
    test_size             = 32 # Related to testing batch_size.
    test_point_batch_size = test_size * test_size * test_size

    # Get coords.
    dima = test_size
    dim = real_size
    aux_x = np.zeros([dima,dima,dima],np.uint8)
    aux_y = np.zeros([dima,dima,dima],np.uint8)
    aux_z = np.zeros([dima,dima,dima],np.uint8)
    multiplier = int(dim/dima)
    multiplier2 = multiplier*multiplier
    multiplier3 = multiplier*multiplier*multiplier
    for i in range(dima):
        for j in range(dima):
            for k in range(dima):
                aux_x[i,j,k] = i*multiplier
                aux_y[i,j,k] = j*multiplier
                aux_z[i,j,k] = k*multiplier
    coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
    for i in range(multiplier):
        for j in range(multiplier):
            for k in range(multiplier):
                coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
                coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
                coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
    coords = (coords+0.5)/dim-0.5
    coords = np.reshape(coords,[multiplier3,test_point_batch_size,3])
    coords = np.concatenate([coords, np.ones([multiplier3,test_point_batch_size,1],np.float32) ],axis=2)
    coords = torch.from_numpy(coords)
    coords = coords.to(device)

    # Instantiate model.
    ef_dim = 32
    p_dim  = 4096 # Number of planes.
    c_dim  = 256  # Number of convexes.
    print("Instantiating model...")
    bsp_net = model.BSPNet(ef_dim, p_dim, c_dim, img_ef_dim=64, z_dim=ef_dim*8)
    bsp_net.to(device)
    bsp_net.eval() # The network must be set to evaluation mode.

    # Load pre-trained model.
    print("Loading model weights loaded from checkpoint...")
    bsp_net.load_state_dict( torch.load(checkpoint_path) )

    # Load the (input) image and convert it to tensor.
    print("Loading image...")
    image = PIL.Image.open(image_path)
    image = IMAGE_TRANSFORM(image)

    # BSP-Net will determine:
    # 1. which planes to use
    # 2. how they are used to form convexes
    # 3. how convexes form the object
    w2 = bsp_net.generator.convex_layer_weights.detach().cpu().numpy()
    multiplier = int(real_size/test_size)
    multiplier2 = multiplier*multiplier
    model_float = np.ones([real_size,real_size,real_size,c_dim], np.float32)
    model_float_combined = np.ones([real_size,real_size,real_size], np.float32)
    batch_view = torch.reshape( image, shape=(1,1,128,128) )
    batch_view = batch_view.to(device)
    print("Predicting planes from image...")
    _, plane_m, _,_ = bsp_net(batch_view, None, None, None, is_training=False)
    print("Predicting partitions...")
    for i in range(multiplier):
        for j in range(multiplier):
            for k in range(multiplier):
                minib = i*multiplier2+j*multiplier+k
                point_coord = coords[minib:minib+1]
                _,_, model_out, model_out_combined = bsp_net(None, None, plane_m, point_coord, is_training=False)
                model_float[aux_x+i,aux_y+j,aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [test_size,test_size,test_size,c_dim])
                model_float_combined[aux_x+i,aux_y+j,aux_z+k] = np.reshape(model_out_combined.detach().cpu().numpy(), [test_size,test_size,test_size])

    # Gather convexes from planes + partitions.
    bsp_convex_list = []
    plane_m = plane_m.detach().cpu().numpy()
    model_float = model_float < 0.01
    model_float_sum = np.sum(model_float, axis=3)
    for i in range(c_dim):
        slice_i = model_float[:,:,:,i]
        if np.max(slice_i) > 0:
            box = []
            for j in range(p_dim):
                if w2[j,i] > 0.01:
                    a = -plane_m[0,0,j]
                    b = -plane_m[0,1,j]
                    c = -plane_m[0,2,j]
                    d = -plane_m[0,3,j]
                    box.append( [a,b,c,d] )
            if len(box) > 0:
                bsp_convex_list.append( np.array(box, np.float32) )

    # Convert bspt to mesh.
    print("Converting list of convexes to watertight mesh...")
    vertices, polygons = bspt.get_mesh_watertight(bsp_convex_list)

    # Convert mesh .ply to .obj
    # Then save mesh on disk.
    print("Saving mesh on disk...")
    outdir_path = os.path.dirname( output_path )
    os.makedirs(outdir_path, exist_ok=True)
    utils.write_ply_polygon(output_path, vertices, polygons)


if __name__ == "__main__":
    import sys
    import ply_to_obj

    image_path      = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_path     = sys.argv[3]

    reconstruct_object(image_path, checkpoint_path, output_path)

    # Convert .ply to .obj
    #obj_path = os.path.splitext(output_path)[0] + '.obj'
    #ply_to_obj.convert( output_path, obj_path )
