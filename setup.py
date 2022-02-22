import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="local2separators-iyassou",
    version="0.0.1",
    author="Iyassou Shimels",
    author_email="s.iyassou@gmail.com",
    description="A library for identifying local 2-separators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iyassou/local2separators",
    project_urls={
        "Bug Tracker": "https://github.com/iyassou/local2separators/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.2",
    package_data={
        # If any package contains *.txt files, include them.
        "": [".txt"],
    }
)
