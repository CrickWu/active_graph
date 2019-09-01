# How to install
The package is based on [pyg](https://rusty1s.github.io/pytorch_geometric/build/html/index.html).
First run the Anaconda environment installation command
``
conda create -n pyg --file requirement_conda.txt -c pytorch 
``
Be sure to set all environment parameters `LD_LIBRARY, CPATH, PATH` correct (see [instructions](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)) before executing the following steps.
Then `source` into `pyg` environment and install the extra pip packages using
``
pip install -r requirement_pip.txt
``

# How to run
`run.sh` shows a simple example. More options could be found using `python active_graph.py --help`

# Data collection and parsing
Please execute the following commands in sequence:
`grid.py`, `dump_csv.py`, `parse_log.py` (get list of metrics), `plot.ipynb`
