from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    with req_path.open() as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="multi_agent_market_research",
    version="1.0.0",
    description="Multi-Agent Market Research System powered by LangChain + Gemini",
    author="Your Name",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "market-research=main:main",   #calls main function in main.py
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
)
