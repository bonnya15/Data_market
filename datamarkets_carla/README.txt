This fold contains the scripts and datasets to run the first synthetic case in paper "Towards Data Markets in Renewable Energy Forecasting"
 
Something I changed meanwhile was the idea in the line "6:" of Algorithm 2 (see paper Towards...): instead of selecting the price randomly from a probability distribution, we can take the mean value of this distribution - this makes the market's price more stable. The old and new definitions of the line "6:" of Algorithm 2 are in lines 85 to 90 of synthetic_run_simulation.py.
