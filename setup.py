from setuptools import setup, find_packages

# Read version from __init__.py
with open('__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read README for the long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='llm_consistency_verifier',
    version=version,
    description='A neural-symbolic verification system for LLM output consistency',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='https://github.com/yourusername/llm_consistency_verifier',
    packages=find_packages(),
    install_requires=[
        'openai>=1.0.0',
        'transformers>=4.30.0',
        'python-dotenv>=1.0.0',
        'sympy>=1.12',
        'z3-solver>=4.12.2',
        'numpy>=1.24.0',
        'pydantic>=2.0.0',
        'pytest>=7.0.0',
        'requests>=2.28.0',
        'tqdm>=4.65.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)
