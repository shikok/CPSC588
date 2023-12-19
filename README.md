# Project
Spatially resolved transcriptomics and proteomics play a pivotal role in deciphering complex biological phenomena, ranging from embryonic development to cancer pathologies. A major obstacle in this field is the integration of varied spatial omics datasets, often hindered by batch effects. These effects, originating from differences in experimental setups, can significantly distort data, thereby concealing actual biological signals. Recent breakthroughs in computational biology, particularly the application of advanced language models such as GPT-3.5 and GPT-2, have shown potential in overcoming these challenges. This study introduces an innovative methodology that synergizes linguistic analysis with spatial biological data by employing language models in conjunction with graph neural network architectures. A comparison aims to asses if this approach can effectively reduce batch effects in spatial omics experiments, paving the way for more accurate and integrative biological insights. The answer is sadly no.

# Docker:
Setting up the repo to work with Docker may be accomplished in the following way. Some variables need to be set and are indicated by [text]
```

# From repo:
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t [docker container name] .
docker run -it -d -p [remote port]:[remote port] --gpus all --mount "type=bind,src=$(pwd),dst=[workdir name]" --workdir [workdir name] [docker container name] /bin/bash
```

To set up and run remote access Jupyter notebooks, one can initiate as follows: 
```
[In Docker session; e.g. post attach]
jupyter notebook password # Set password in session
jupyter notebook --ip 0.0.0.0 --port [remote port] --no-browser # Replace remote port with any port (4-digit number, e.g. 1234) 
# Docker 

[From non-remote machine]
ssh -L [remote port]:localhost:[local port] [user]@[server] # Replace variables with appropriate terms
localhost:[local port] # In browser
```

An example session would be:
```
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t shay:sorbet .
docker run -it -d -p [remote port]:[remote port] --gpus all --mount "type=bind,src=$(pwd),dst=/matrix_completion" --workdir /Project shay:sorbet /bin/bash

docker attach [Docker assigned name]

[In Docker Session]
jupyter notebook password # Set password via interactive commands
jupyter notebook --ip 0.0.0.0 --port 7667 --no-browser

[Locally]
ssh -L 7667:localhost:1234 joe@abel.med.yale.edu
[In Browser]
localhost:1234 # Will need to enter password set in Docker session 
```
# Notebook to Replicate Results
main_notebook.ipynb

# Data
available on request

## Folders
data_handling - preprocessing of data 
learning - learning related code