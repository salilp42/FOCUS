from setuptools import setup, find_packages

setup(
    name="saccadenet",
    version="1.0.0",
    description="Bio-Inspired Deep Learning Architecture for Medical Image Analysis",
    author="[Author List]",
    author_email="[Email]",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pillow>=8.3.0',
        'tqdm>=4.62.0',
        'medmnist>=2.1.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'visualization': ['torchcam>=0.3.0'],
    },
    python_requires='>=3.8',
)
