"""
Copies the deccorelations kernels in the virtual environment packages.
"""

import os
import shutil
import site

source_path = os.path.abspath("../pygfotoolbox/pyGFOToolbox/processing/filter/kernels")

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
destination_path = os.path.join(site_packages, "pyGFOToolbox", "processing", "filter", "kernels")
os.makedirs(os.path.dirname(destination_path), exist_ok=True)
shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
