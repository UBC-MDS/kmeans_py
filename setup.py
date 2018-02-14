from distutils.core import setup

setup(
    name='kmeans_py',
    version='1.0',
    author='J. Random Hacker',
    author_email='jrh@example.com',
    packages=['kmeans_py', 'kmeans_py.test'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Kmeans clustering',
    long_description=open('README.txt').read(),
    #install_requires=[
    #    "Django >= 1.1.1",
    #    "caldav == 0.1.4",
    #],
)
