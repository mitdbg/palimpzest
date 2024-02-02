from setuptools import setup, find_packages

setup(
    # Basic package information:
    name='Palimpzest',
    version='0.1.0',
    author='MIT DSG Semantic Management Lab',
    author_email='michjc@csail.mit.edu',

    # Package description:
    description='Palimpzest is the next generation of extraction and document data management',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mikecafarella/Palimpzest',

    # Specify package directories:
    package_dir={'': 'src'},  # if your code is in a src directory
    packages=find_packages(where='src'),

    # Runtime dependencies (these will be installed along with the package):
    install_requires=[
        'numpy>=1.19',
        'requests>=2.25',
        'pandas>=2.2.0',
        'pyarrow>=13.0.0',
        'fastapi~=0.100.0',
        'openai>=1.0',
        'dspy-ai',
        # Add other dependencies as needed
    ],

    # Additional metadata for PyPI:
    classifiers=[
        'Development Status :: 4 - Beta',  # Change as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Change as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',  # Specify versions you support
        # Add more classifiers as appropriate
    ],
    keywords='extraction llm tools document search integration',  # Add keywords relevant to your package

    # Include additional files into the package:
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.rst', '*.md']
    #    'palimpzest': ['data/*.data'],
        # Include other non-python data files your package needs here
    }

    # Entry points for command line tools:
    #entry_points={
    #    'console_scripts': [
    #        'yourscript = your_package.module:main_function',
    #        # Add more console scripts as needed
    #    ],
    #},
)

