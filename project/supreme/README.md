# SUPREME

## Local install and Setup

1. Create environment with pytorch installed. Use `cpuonly` version if you don't have an nvidia gpu locally:

   ```shell script
   conda create -p env python=3.8.10 -y
   conda activate ./env
   ```

# Install pre-commit hook

```shell script
pip install pre-commit
```

# Install the python dependencies

```shell script
pip install -r requirements.txt
pip install -r requirements-dev.txt
```


# Add a pre-commit configuration
```shell script
pre-commit install

```
