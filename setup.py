from setuptools import setup

setup(
    name='Gaussian Search',
    version='0.1.0',
    description='Gaussian process adaptive grid search.',
    url='https://github.com/Richard-Sti/GaussianSearch',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0 License',
    packages=['gaussian_search'],
    install_requires=['scipy',
                      'numpy',
                      'scikit-learn',
                      'joblib',
                      'dynesty'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
