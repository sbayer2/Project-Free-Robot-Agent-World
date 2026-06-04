"""Data-generation subpackage.

Deliberately does *not* import ``generate_blender`` (which requires ``bpy``) or
heavy optional deps at package import time, so ``import pseudomarble.data`` works
in plain CPython / CI. Import the Blender entry point explicitly when running
inside Blender:

    blender --background --python -m pseudomarble.data.generate_blender -- ...
"""
