"""Setup script for rhasspy-wake-raven package"""
import sys
from pathlib import Path

import numpy as np
import setuptools
from setuptools.command.build_ext import build_ext

this_dir = Path(__file__).parent

# -----------------------------------------------------------------------------

# Load README in as long description
long_description: str = ""
readme_path = this_dir / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text()

requirements_path = this_dir / "requirements.txt"
with open(requirements_path, "r") as requirements_file:
    requirements = requirements_file.read().splitlines()

version_path = this_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc", "/O2", "/std:c++11", "/W4"],
        "unix": ["-O3", "-std=c++11", "-Wextra", "-Wall", "-Wconversion", "-g0"],
    }
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        build_ext.build_extensions(self)


ext_modules = [
    setuptools.Extension(
        name="rhasspywake_raven.dtw",
        sources=["rhasspywake_raven/dtw.c"],
        include_dirs=[np.get_include()],
    )
]

setuptools.setup(
    name="rhasspy-wake-raven",
    version=version,
    description="Simple wake word detection using audio templates",
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/rhasspy-wake-raven",
    packages=setuptools.find_packages(),
    package_data={"rhasspywake_raven": ["py.typed"]},
    install_requires=requirements,
    cmdclass={"build_ext": BuildExt},
    ext_modules=ext_modules,
    entry_points={
        "console_scripts": ["rhasspy-wake-raven = rhasspywake_raven.__main__:main"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
