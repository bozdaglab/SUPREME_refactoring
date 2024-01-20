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
#pip install -e lib/supreme
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```


# Add a pre-commit configuration
```shell script
pre-commit install

```
### run this to clean docker espace sudo docker container prune -f && sudo docker image prune -f docker system prune -a

### How to run.
Organize your multi-omics data by placing it in the data/sample_data/raw directory. Modify the prepare_data function within the pre_process_data.py module, specifically change lines 60 to 92. Then, execute main.py.