from setuptools import setup

setup(
    name="LLMedGenie",
    packages=["LLMedGenie"],
    package_dir={'LLMedGenie': 'LLMedGenie'},
    version="0.3.0",
    author="Tung Tran",
    description="A LLM-based medical transcript generator",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tungsontran/llm-transcript-generator/",
    license="MIT",
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
