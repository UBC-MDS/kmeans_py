from distutils.core import setup

setup(
    name='kmeans_py',
    version='2.0',
    author='Charley Carriero, Johannes Harmse, Bradley Pick',
    author_email='NA',
    packages=['kmeans_py'],
    url='https://github.com/UBC-MDS/kmeans_py',
    license='LICENSE.md',
    description='Kmeans initialization, clustering, and plotting',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn']
)
