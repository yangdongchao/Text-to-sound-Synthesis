#### How to build fast loader data
In this part, we introduce how to build fast loader. The core idea is that split all of the data to each GPU, and then save each batch in advance. <br/>

**save_32_gpu.py** includes the code of how to split data <br/>
**json_32gpu.py** includes how to build json files for each GPU.
##### We will give more details about how to build the fast loader in the future.