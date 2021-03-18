import os
import subprocess
import sys
import time


# Data paths.
data_path                      = os.path.realpath( sys.argv[1] )
input_photos_dir_path          = os.path.join( data_path, "0_Input_photos/" )
segmentation_dir_path          = os.path.join( data_path, '1_Segmentation/' )
mask_extraction_dir_path       = os.path.join( data_path, '2_Mask_Extraction/' )
object_reconstruction_dir_path = os.path.join( data_path, '3_Object_Reconstruction/' )
remeshing_dir_path             = os.path.join( data_path, '4_Remeshing/' )
rendering_generation_dir_path  = os.path.join( data_path, '5_Rendering_Generation/' )
rotation_estimation_dir_path   = os.path.join( data_path, '6_Rotation_Estimation/' )
object_rotation_dir_path       = os.path.join( data_path, '7_Object_Rotation/' )

# Data -> Segmentation globals.
segmentation_config_path = os.path.join( data_path, '../../Tests/Mseg/eval_config.yaml' )
segmentation_model_path  = os.path.join( data_path, '../../Tests/Mseg//mseg-3m.pth' )
segmentation_device      = 'cuda'

# Data -> BSP globals.
bsp_weights_path = os.path.join( data_path, '../../Tests/BSP-Net/BSP_SVR.model-1000.pth' )

# Data -> PoseFromShape globals
pose_from_shape_view_num   = "--view_num=12",
pose_from_shape_shape      = "--shape=MultiView",
pose_from_shape_model_path = "--model=" + os.path.join( data_path, '../../Tests/PoseFromShape/ObjectNet3D.pth' )

#print("root: ", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Program root and script paths. 
program_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

segmentation_script_path          = os.path.join( program_root, 'Mseg/Mseg/segment_image.py' )
binary_mask_creation_script_path  = os.path.join( program_root, 'ObjectMasking/ObjectMasking/create_binary_mask.py' )
mask_extraction_script_path       = os.path.join( program_root, 'ObjectMasking/ObjectMasking/object_masking.py' )
object_reconstruction_script_path = os.path.join( program_root, 'BSP-Net/BSPNet/reconstruct_object.py' )
remeshing_script_path             = os.path.join( program_root, 'Remeshing/remeshing.py' )
rendering_generation_script_path  = os.path.join( program_root, 'Rendering/rendering.py' )
pose_estimation_script_path       = os.path.join( program_root, 'PoseFromShape/PoseFromShape/inference.py' )
object_rotation_script_path       = os.path.join( program_root, 'ObjectRotation/rotating.py')


def reconstruct_object(filepath):
    filename = os.path.splitext( os.path.basename(filepath) )[0]
    print("filename: ", filename)
    print("filepath: ", filepath)

    # Segmentation.
    print("---------- Processing Semantic Segmentation...")
    grayscale_image_path = os.path.join( segmentation_dir_path, '%s_grayscale.png' % filename )
    cmd = [
        sys.executable,
        segmentation_script_path,
        filepath,
        segmentation_config_path,
        segmentation_device,
        grayscale_image_path
    ]
    subprocess.call(cmd)
    print("---------- Semantic Segmentation Done")
    
    # Binary Mask Creation.
    binary_mask_path = os.path.join( mask_extraction_dir_path, '%s_binary.png' % filename )
    cmd = [
        sys.executable,
        binary_mask_creation_script_path,
        grayscale_image_path,
        binary_mask_path
    ]
    subprocess.call(cmd)
    print("---------- Mask Creation Done")

    # Mask Extraction
    extracted_object_path = os.path.join( mask_extraction_dir_path, '%s_extracted.png' % filename )
    cmd = [
        sys.executable,
        mask_extraction_script_path,
        filepath,
        binary_mask_path,
        extracted_object_path
    ]
    subprocess.call(cmd)
    print("---------- Mask Extraction Done")

    # Object Reconstruction.
    reconstructed_object_ply_path = os.path.join( object_reconstruction_dir_path, '%s.ply' % filename )
    cmd = [
        sys.executable,
        object_reconstruction_script_path,
        extracted_object_path,
        bsp_weights_path,
        reconstructed_object_ply_path
    ]
    subprocess.call(cmd)
    print("---------- Object Reconstruction Done")
    
    # Remeshing.
    cmd = [
        sys.executable,
        remeshing_script_path,
        object_reconstruction_dir_path
    ]
    subprocess.call(cmd)
    print("---------- Remeshing Done")

    # Rendering Generation.
    cmd = [
        sys.executable,
        rendering_generation_script_path,
        remeshing_dir_path
    ]
    subprocess.call(cmd)
    print("---------- Rendering Generation Done")

    # PoseFromShape.
    rotation_txt_path = os.path.join( rotation_estimation_dir_path, '%s.txt' % filename )
    cmd = [
        sys.executable,
        pose_estimation_script_path,
        pose_from_shape_view_num,
        pose_from_shape_shape,
        pose_from_shape_model_path,
        "--image_path=" + extracted_object_path,
        "--render_path=" + os.path.join( rendering_generation_dir_path, filename ),
        "--out_path=" + rotation_txt_path
    ]
    subprocess.call(cmd)
    print("---------- Pose Estimation Done")

    # Rotating.
    cmd = [
        sys.executable,
        object_rotation_script_path,
        remeshing_dir_path
    ]
    subprocess.call(cmd)
    print("---------- Rotating Done")

    return # ob


def send_object(ob):
    pass


# Crea connessione TCP.
#tcp_client = ...

# Crea cartelle.
os.makedirs(input_photos_dir_path, exist_ok=True)
os.makedirs(segmentation_dir_path, exist_ok=True)
os.makedirs(mask_extraction_dir_path, exist_ok=True)
os.makedirs(object_reconstruction_dir_path, exist_ok=True)
os.makedirs(remeshing_dir_path, exist_ok=True)
os.makedirs(rendering_generation_dir_path, exist_ok=True)
os.makedirs(rotation_estimation_dir_path, exist_ok=True)
os.makedirs(object_rotation_dir_path, exist_ok=True)

while True:
    # Riceve dati da client.
    #tcp_client.receive(...)

    # Controlla se ci sono dati da processare.
    filenames = os.listdir( input_photos_dir_path )
    if len(filenames) == 0:
        time.sleep(.1)
        continue

    # Processa dati (segmentazione, mesh, etc.).
    for filename in filenames:
        filepath = os.path.join( input_photos_dir_path, filename )
        # ob = reconstruct_object(filepath)
        reconstruct_object(filepath)

        #send_object(ob)

        # Rimuovi foto per non processarla 2 volte.
        #os.remove(filepath)

    break # TEMP: per la demo, esci subito dal ciclo while
