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
pip install -r project/supreme/requirements.txt
pip install -r project/supreme/requirements-dev.txt
pip install -e lib/supreme
```


# Add a pre-commit configuration
```shell script
pre-commit install

```
