from setuptools import setup, find_packages

setup(
    name='gliner-finetune',
    version='0.0.5',
    author='William J.B. Mattingly',
    description='A library to create synthetic data with OpenAI and train a GLiNER model on that data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wjbmattingly/gliner-finetune',
    packages=find_packages(),
    install_requires=[
        'gliner',
        'openai',
        'spacy',
        'scipy==1.12',
        'torch',
        'transformers',
        'tqdm',
        'python-dotenv'
    ],
    python_requires='>=3.7, <=3.11',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # Assuming MIT License, replace if different
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
