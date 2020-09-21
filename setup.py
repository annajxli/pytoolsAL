import os
from setuptools import setup

# read __version__
filename = 'pytoolsAL/_version.py';
exec (compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pytoolsAL",
    version=__version__,
    author="Anna Li",
    author_email="annajxli@gmail.com",
    description="misc personal Python utilities",
    license="none",
    keywords="",
    url="",
    packages=['pytoolsAL'],
    package_dir={'pytoolsAL': 'pytoolsAL'},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ], install_requires=['matplotlib', 'scikit-image', 'tifffile', 'numpy', 'imageio']
)
