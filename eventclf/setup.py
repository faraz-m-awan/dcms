import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eventclf",
    version="0.0.5",
    author="Cris Vini",
    author_email="cristina.muntean@isti.cnr.it",
    description="An event classification package by Vinicius Monteiro de Lira",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "eventclf"},
    # packages=setuptools.find_packages(where="eventclf"),
    packages=["eventclf", "eventclf.data"],
    python_requires=">=3.6",
    include_package_data=True
)