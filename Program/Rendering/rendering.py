import os
import subprocess


BLENDER_RELPATH = '../../External Programs/blender-2.83.12-windows64/blender.exe'


script_dir_path      = os.path.dirname( os.path.realpath(__file__) )
blender_path         = os.path.realpath( os.path.join(script_dir_path, BLENDER_RELPATH ) )
rendering_blend_path = os.path.join( script_dir_path, 'Rendering.blend' )


def render_objects(dir_path):
    cmd = [
        blender_path,
        '--background',
        rendering_blend_path,
        '--python-text',
        'render_objects.py'
    ]
    subprocess.call(cmd, cwd=dir_path)


if __name__ == '__main__':
    import sys

    dir_path = sys.argv[1]

    render_objects(dir_path)
