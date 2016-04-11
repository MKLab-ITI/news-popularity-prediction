__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='news-popularity-prediction',
    version='0.1.1',
    author='Georgios Rizos',
    author_email='georgerizos@iti.gr',
    packages=['news_popularity_prediction',
              'news_popularity_prediction.datautil',
              'news_popularity_prediction.entry_points',
              'news_popularity_prediction.entry_points.snow_2016_workshop',
              'news_popularity_prediction.features',
              'news_popularity_prediction.learning',
              'news_popularity_prediction.visualization'],
    url='https://github.com/MKLab-ITI/news-popularity-prediction',
    license='Apache',
    description='A set of methods that predict the future values of popularity indices for news posts using a variety of features.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    keywords="news-popularity-prediction online-discussion-analysis Reveal-FP7",
    # entry_points={
    #     'console_scripts': ['run_all_experiments=news_popularity_prediction.entry_points.snow_2016_workshop.run_all_experiments:main'],
    # },
    install_requires=open("requirements.txt").read().split("\n")
)
