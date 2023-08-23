# WS-DMF
PyTorch implementation for our paper (submitted):    

"Deep matched filtering for retinal vessel segmentation"

The full material will be made available once the manuscript is accepted!!!

## Project

```
Project for WS-DMF
    ├── data 
        ├── eyeset.py (dataset & dataloader & data pre-processing)  
        └── ...  
    ├── nets 
        ├── modules/activation.py (activation functions)  
        ├── modules/attention.py (attention modules)  
        ├── conv.py (convolution layers)  
        ├── dmfu.py (deep matched filtering)  
        ├── lunet.py (lightweight UNet)  
        ├── rot.py (APC layer related OAL loss function)  
        └── ...   
    ├── utils  
        ├── loss.py (loss function)  
        ├── optim.py (optimizer for training)  
        └── ...  
    ├── build.py (implementation for WS-DMF)  
    ├── grad.py (implementation for backgrading)  
    ├── loop.py (implementation for training)  
    ├── main.py (implementation for main function)  
    └── ...   
    ├── onnx (Pytorch trained weights)  
        └── infer.py (infer fundus images for segmentation with *.onnx weights)  
        └── ...  
    ├── results (segmentation results)  
        └── ...  (segmentation results for popular datasets)
        └── ...  (segmentation results for cross-dataset-validation)
```




## Contact
For any questions, please contact me. 
And my e-mails are 
-   tyb311@qq.com
-   tyb@std.uestc.edu.cn
