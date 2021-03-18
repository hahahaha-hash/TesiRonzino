# BSP-Net
PyTorch implementation of [BSP-Net: Generating Compact Meshes via Binary Space Partitioning](https://arxiv.org/abs/1911.06971).
This implementation is based on [this repository](https://github.com/czq142857/BSP-NET-pytorch).


## Other Implementations
- [original](https://github.com/czq142857/BSP-NET-original)
- [TensorFlow 1.15 (Static graph)](https://github.com/czq142857/BSP-NET-tf1)
- [TensorFlow 2.0 (Eager execution)](https://github.com/czq142857/BSP-NET-tf2)
- [PyTorch 1.2](https://github.com/czq142857/BSP-NET-pytorch)


## Dependencies
Requirements:
- numpy==1.19.1
- pillow==7.2.0
- torch==1.6.0
- torchvision==0.7.0

This code was tested on a machine with Windows 10 and Python 3.6.


## Usage
First, you have to clone the repository and install the dependencies.
```
git clone [repository_link]
cd [dir_name]
pip install -r requirements.txt
```

To reconstruct an object from a single image.
```
python reconstruct_object [image_path] [checkpoint_path] [output_path]
```


## License
This project is licensed under the terms of the MIT license (see LICENSE for details).
