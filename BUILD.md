

**Python install**   
```bash
~ python3 -m pip uninstall paddlepaddle
~ python3 -m pip install paddlepaddle==1.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Inference Build**   
```bash
~ export PADDLE_ROOT=/Users/taozj/repos/machine_learning/Paddle/inference_root
~ cd build
~ cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT -DCMAKE_BUILD_TYPE=Debug -DWITH_PYTHON=OFF -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_NGRAPH=OFF ..
~ make -j4
~ make inference_lib_dist
```

**Next**
Follow up this repos ;-)   
