# arctopus/setup.py
from setuptools import setup, find_packages

setup(
    name='arctopus_trainer',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'), # This will find your 'src' package
    install_requires=[ # These should largely mirror your requirements.txt
        'torch',
        'transformers',
        'numpy',
        'bitsandbytes', # Needed for 4-bit quantization
        'accelerate',   # Needed for 4-bit quantization and distributed training
        'pandas',       # Potentially useful for data handling
        'python-json-logger',
        'datasets',
        'gcsfs'
        # Add any other libraries your common.py or training code uses
    ],
    python_requires='>=3.9', # Specify Python version compatibility
)