"""Build Cython extensions in-place.

Usage:
    cd src/models/speech-models
    python build_cython.py
"""

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension

ext = Extension(
    name="speech_models.modules.others.tts.monotonic_align_core",
    sources=[
        "speech_models/modules/others/tts/monotonic_align_core.pyx",
    ],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

ext_modules = cythonize([ext], language_level=3)

dist = Distribution({"ext_modules": ext_modules})
dist.parse_config_files()

cmd = dist.get_command_obj("build_ext")
cmd.inplace = True
cmd.ensure_finalized()
cmd.run()
