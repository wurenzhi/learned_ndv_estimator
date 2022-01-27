import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="estndv",
    version="0.0.2",
    author="Renzhi Wu",
    author_email="renzhiwu@gatech.edu",
    description="Learned sample-based estimator for number of distinct values.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wurenzhi/learned_ndv_estimator",
    project_urls={
        "Bug Tracker": "https://github.com/wurenzhi/learned_ndv_estimator/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy'],
    include_package_data=True
)