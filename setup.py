from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="grib2_to_csv",
    version="0.1.0",
    license="MIT",
    description='GRIB2ファイルをCSVファイルへ変換します',
    author="ArcClimate Development Team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    install_requires=_requires_from_file('requirements.txt'),
    entry_points= {
        'console_scripts': ['grib2_to_csv=grib2_to_csv.grib2_to_csv:main']
    }
)