import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="drughive",
    version="0.0.1",
    author="Jesse Weller and Remo Rohs",
    author_email="wellerj@usc.edu",
    description="DrugHIVE: Structure-based drug design with a deep hierarchical generative model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/jssweller/DrugHIVE",
    packages=['drughive'],
    classifiers=[
	"Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9',
)