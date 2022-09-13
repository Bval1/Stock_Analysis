# First time setup:
conda create --name myenv
conda init cmd.exe

# Install packages   
conda install numpy
conda install pandas_datareader
conda install matplotlib
shift+ctrl+p > select pyton interpreter
conda install -c plotly plotly=5.9.0

# Activate enviornment
conda activate myenv (stock_analysis for this project)

# See all conda enviornments: 
conda env list   
