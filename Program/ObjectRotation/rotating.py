import os
import subprocess


BLENDER_RELPATH = '../../External Programs/blender-2.83.12-windows64/blender.exe'


script_dir_path      = os.path.dirname( os.path.realpath(__file__) )
blender_path         = os.path.realpath( os.path.join(script_dir_path, BLENDER_RELPATH ) )
rotating_blend_path  = os.path.join( script_dir_path, 'Rotating.blend' )


def rotate_objects(dir_path):
    cmd = [
        blender_path,
        '--background',
        rotating_blend_path,
        '--python-text',
        'rotate_objects.py'
    ]
    subprocess.call(cmd, cwd=dir_path)


if __name__ == '__main__':
    import sys

    dir_path = sys.argv[1]

    rotate_objects(dir_path)
