# Overview

Yolov3 model for <a href="https://dac.com/content/2020-system-design-contest">DAC SDC 2020</a>.

这个比赛大概就是要对无人机拍摄的照片中进行目标检测，要求在一块Ultra96的FPGA板子上跑，测试指标有iou、fps和功耗。因为现在新冠疫情回不去学校，所以没有板子，所以我已经弃疗了。这个repo是训练模型的代码。

需要做的：
- 模型压缩
- 现在把resnet_18里的第一个7x7卷积换成了3x3卷积（为了硬件上少占用资源），先看一看效果
- ~~改成depthwise conv~~
- ~~anchor大小cluster（kmeans）~~
  - 先用sklearn里的kmeans得到欧几里得距离的centers
  - 将上面的centers作为初值，用IOU再跑一个kmeans
  - 得到的新的anchors（是相对于$640\times352$大小的）: 
  ```python
  "anchors": [[[117.92, 57.98], [56.57, 139.19], [163.46, 132.17]],
              [[48.65, 45.88], [35.41, 81.16], [75.67, 76.64]],
              [[12.90, 26.01], [29.35, 29.03], [24.03, 52.87]]],
  ```
- ~~Yolo loss需要稍微改一下~~
  - 修改noobj_mask的计算
  - w和h的loss乘上(2-gt.w*gt.h)，现在乘的是这玩意的平方
- 其他的backbone？以及可以考虑先在其他的数据集训练backbone，因为没有depthwise conv的预训练模型

# Usage

## 0. 配置环境

没有`requirements.txt`，因为我没用虚拟环境，一生成会出来一大堆。运行一下看哪个没有装哪个。

- 服务器系统：ubuntu 16.04
- pytorch版本：1.1.0
- cuda版本：9.0
- python版本：Python 3.6.6 |Anaconda, Inc.

<details>
<summary>conda list</summary>
<pre>
$ conda list
# packages in environment at /home/dingxy/anaconda3:
#
# Name                    Version                   Build  Channel
_ipyw_jlab_nb_ext_conf    0.1.0            py36he11e457_0  
_libgcc_mutex             0.1                        main  
alabaster                 0.7.10           py36h306e16b_0  
anaconda                  5.1.0                    py36_2  
anaconda-client           1.6.9                    py36_0  
anaconda-navigator        1.7.0                    py36_0  
anaconda-project          0.8.2            py36h44fb852_0  
asn1crypto                0.24.0                   py36_0  
astroid                   1.6.1                    py36_0  
astropy                   2.0.3            py36h14c3975_0  
attrs                     17.4.0                   py36_0  
babel                     2.5.3                    py36_0  
backports                 1.0              py36hfa02d7e_1  
backports.shutil_get_terminal_size 1.0.0            py36hfea85ff_2  
beautifulsoup4            4.6.0            py36h49b8c8c_1  
bitarray                  0.8.1            py36h14c3975_1  
bkcharts                  0.2              py36h735825a_0  
blas                      1.0                         mkl  
blaze                     0.11.3           py36h4e06776_0  
bleach                    2.1.2                    py36_0  
bokeh                     0.12.13          py36h2f9c1c0_0  
boto                      2.48.0           py36h6e4cd66_1  
bottleneck                1.2.1            py36haac1ea0_0  
bzip2                     1.0.6                h9a117a8_4  
ca-certificates           2020.1.1                      0  
cairo                     1.14.12              h77bcde2_0  
certifi                   2019.11.28               py36_0  
cffi                      1.13.2           py36h2e261b9_0  
chardet                   3.0.4            py36h0f667ec_1  
click                     6.7              py36h5253387_0  
cloudpickle               0.5.2                    py36_1  
clyent                    1.2.2            py36h7e57e65_1  
colorama                  0.3.9            py36h489cec4_0  
conda                     4.4.10                   py36_0  
conda-build               3.4.1                    py36_0  
conda-env                 2.6.0                h36134e3_1  
conda-verify              2.0.0            py36h98955d8_0  
contextlib2               0.5.5            py36h6c84a62_0  
cryptography              2.1.4            py36hd09be54_0  
cudatoolkit               9.0                  h13b8566_0  
curl                      7.58.0               h84994c4_0  
cycler                    0.10.0           py36h93f1223_0  
cython                    0.29.14          py36he6710b0_0  
cytoolz                   0.9.0            py36h14c3975_0  
dask                      0.16.1                   py36_0  
dask-core                 0.16.1                   py36_0  
datashape                 0.5.4            py36h3ad6b5c_0  
dbus                      1.12.2               hc3f9b76_1  
decorator                 4.2.1                    py36_0  
distributed               1.20.2                   py36_0  
docutils                  0.14             py36hb0f60f5_0  
entrypoints               0.2.3            py36h1aec115_2  
et_xmlfile                1.0.1            py36hd6bccc3_0  
expat                     2.2.5                he0dffb1_0  
fastcache                 1.0.2            py36h14c3975_2  
ffmpeg                    4.0                  h04d0a96_0  
filelock                  2.0.13           py36h646ffb5_0  
flask                     0.12.2           py36hb24657c_0  
flask-cors                3.0.3            py36h2d857d3_0  
fontconfig                2.12.6               h49f89f6_0  
freetype                  2.8                  hab7d2ae_1  
get_terminal_size         1.0.0                haa9412d_0  
gevent                    1.2.2            py36h2fe25dc_0  
glib                      2.53.6               h5d9569c_2  
glob2                     0.6              py36he249c77_0  
gmp                       6.1.2                h6c8ec71_1  
gmpy2                     2.0.8            py36hc8893dd_2  
graphite2                 1.3.10               hf63cedd_1  
greenlet                  0.4.12           py36h2d503a6_0  
gst-plugins-base          1.12.4               h33fb286_0  
gstreamer                 1.12.4               hb53b477_0  
h5py                      2.7.1            py36h3585f63_0  
harfbuzz                  1.7.6                hc5b324e_0  
hdf5                      1.10.2               hba1933b_1  
heapdict                  1.0.0                    py36_2  
html5lib                  1.0.1            py36h2f9c1c0_0  
icu                       58.2                 h9c2bf20_1  
idna                      2.6              py36h82fb2a8_1  
imageio                   2.2.0            py36he555465_0  
imagesize                 0.7.1            py36h52d8127_0  
intel-openmp              2018.0.0             hc7b2577_8  
ipykernel                 4.8.0                    py36_0  
ipython                   6.2.1            py36h88c514a_1  
ipython_genutils          0.2.0            py36hb52b0d5_0  
ipywidgets                7.1.1                    py36_0  
isort                     4.2.15           py36had401c0_0  
itsdangerous              0.24             py36h93cc618_1  
jasper                    1.900.1              hd497a04_4  
jbig                      2.1                  hdba287a_0  
jdcal                     1.3              py36h4c697fb_0  
jedi                      0.11.1                   py36_0  
jinja2                    2.10             py36ha16c418_0  
jpeg                      9b                   h024ee3a_2  
jsonschema                2.6.0            py36h006f8b5_0  
jupyter                   1.0.0                    py36_4  
jupyter_client            5.2.2                    py36_0  
jupyter_console           5.2.0            py36he59e554_1  
jupyter_core              4.4.0            py36h7c827e3_0  
jupyterlab                0.31.5                   py36_0  
jupyterlab_launcher       0.10.2                   py36_0  
lazy-object-proxy         1.3.1            py36h10fcdad_0  
libcurl                   7.58.0               h1ad7b7a_0  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.2.0                h9f7466a_2  
libopencv                 3.4.1                h1a3b859_1  
libopus                   1.3                  h7b6447c_0  
libpng                    1.6.37               hbc83047_0  
libprotobuf               3.5.2                h6f1eeef_0  
libsodium                 1.0.15               hf101ebd_0  
libssh2                   1.8.0                h9cfc8f7_4  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.0.9                he85c1e1_1  
libtool                   2.4.6                h544aabb_3  
libvpx                    1.7.0                h439df22_0  
libxcb                    1.12                 hcd93eb1_4  
libxml2                   2.9.7                h26e45fe_0  
libxslt                   1.1.32               h1312cb7_0  
llvmlite                  0.21.0           py36ha241eea_0  
locket                    0.2.0            py36h787c0ad_1  
lxml                      4.1.1            py36hf71bdeb_1  
lzo                       2.10                 h49e0be7_2  
markupsafe                1.0              py36hd9260cd_1  
matplotlib                2.1.2            py36h0e671d2_0  
mccabe                    0.6.1            py36h5ad9710_1  
mistune                   0.8.3                    py36_0  
mkl                       2018.0.1             h19d6760_4  
mkl-service               1.1.2            py36h17a0993_4  
mpc                       1.0.3                hec55b23_5  
mpfr                      3.1.5                h11a74b3_2  
mpmath                    1.0.0            py36hfeacd6b_2  
msgpack-python            0.5.1            py36h6bb024c_0  
multipledispatch          0.4.9            py36h41da3fb_0  
navigator-updater         0.1.0            py36h14770f7_0  
nbconvert                 5.3.1            py36hb41ffb7_0  
nbformat                  4.4.0            py36h31c9010_0  
ncurses                   6.1                  hf484d3e_0  
networkx                  2.1                      py36_0  
ninja                     1.8.2            py36h6bb024c_1  
nltk                      3.2.5            py36h7532b22_0  
nose                      1.3.7            py36hcdf7029_2  
notebook                  5.4.0                    py36_0  
numba                     0.36.2          np114py36hc6662d5_0  
numexpr                   2.6.4            py36hc4a3f9a_0  
numpy                     1.14.2           py36hdbf6ddf_0  
numpydoc                  0.7.0            py36h18f165f_0  
odo                       0.5.1            py36h90ed295_0  
olefile                   0.46                       py_0  
opencv                    3.4.1            py36h6fd60c2_2  
openpyxl                  2.4.10                   py36_0  
openssl                   1.0.2u               h7b6447c_0  
packaging                 16.8             py36ha668100_1  
pandas                    0.22.0           py36hf484d3e_0  
pandoc                    1.19.2.1             hea2e7c5_1  
pandocfilters             1.4.2            py36ha6701b7_1  
pango                     1.41.0               hd475d92_0  
parso                     0.1.1            py36h35f843b_0  
partd                     0.3.8            py36h36fd896_0  
patchelf                  0.9                  hf79760b_2  
path.py                   10.5             py36h55ceabb_0  
pathlib2                  2.3.0            py36h49efa8e_0  
patsy                     0.5.0                    py36_0  
pcre                      8.41                 hc27e229_1  
pep8                      1.7.1                    py36_0  
pexpect                   4.3.1                    py36_0  
pickleshare               0.7.4            py36h63277f8_0  
pillow                    5.1.0            py36h3deb7b8_0  
pip                       19.3.1                   py36_0  
pixman                    0.34.0               hceecf20_3  
pkginfo                   1.4.1            py36h215d178_1  
pluggy                    0.6.0            py36hb689045_0  
ply                       3.10             py36hed35086_0  
prompt_toolkit            1.0.15           py36h17d85b1_0  
psutil                    5.4.3            py36h14c3975_0  
ptyprocess                0.5.2            py36h69acd42_0  
py                        1.5.2            py36h29bf505_0  
py-opencv                 3.4.1            py36h0676e08_1  
pycodestyle               2.3.1            py36hf609f19_0  
pycosat                   0.6.3            py36h0a5515d_0  
pycparser                 2.19                       py_0  
pycrypto                  2.6.1            py36h14c3975_7  
pycurl                    7.43.0.1         py36hb7f436b_0  
pyflakes                  1.6.0            py36h7bd6a15_0  
pygments                  2.2.0            py36h0d3125c_0  
pylint                    1.8.2                    py36_0  
pyodbc                    4.0.22           py36hf484d3e_0  
pyopenssl                 17.5.0           py36h20ba746_0  
pyparsing                 2.2.0            py36hee85983_1  
pyqt                      5.6.0            py36h0386399_5  
pysocks                   1.6.7            py36hd97a5b1_1  
pytables                  3.4.2            py36h3b5282a_2  
pytest                    3.3.2                    py36_0  
python                    3.6.6                hc3d631a_0  
python-dateutil           2.6.1            py36h88d3b88_1  
pytorch                   1.1.0           py3.6_cuda9.0.176_cudnn7.5.1_0    pytorch
pytz                      2017.3           py36h63b9c63_0  
pywavelets                0.5.2            py36he602eb0_0  
pyyaml                    3.12             py36hafb9ca4_1  
pyzmq                     16.0.3           py36he2533c7_0  
qt                        5.6.2               h974d657_12  
qtawesome                 0.4.4            py36h609ed8c_0  
qtconsole                 4.3.1            py36h8f73b5b_0  
qtpy                      1.3.1            py36h3691cc8_0  
readline                  7.0                  h7b6447c_5  
requests                  2.18.4           py36he2e5f8d_1  
rope                      0.10.7           py36h147e2ec_0  
ruamel_yaml               0.15.35          py36h14c3975_1  
scikit-image              0.13.1           py36h14c3975_1  
scikit-learn              0.19.1           py36h7aa7ec6_0  
scipy                     1.0.0            py36hbf646e7_0  
seaborn                   0.8.1            py36hfad7ec4_0  
send2trash                1.4.2                    py36_0  
setuptools                44.0.0                   py36_0  
simplegeneric             0.8.1                    py36_2  
singledispatch            3.4.0.3          py36h7a266c3_0  
sip                       4.18.1           py36h51ed4ed_2  
six                       1.13.0                   py36_0  
snowballstemmer           1.2.1            py36h6febd40_0  
sortedcollections         0.5.3            py36h3c761f9_0  
sortedcontainers          1.5.9                    py36_0  
sphinx                    1.6.6                    py36_0  
sphinxcontrib             1.0              py36h6d0f590_1  
sphinxcontrib-websupport  1.0.1            py36hb5cb234_1  
spyder                    3.2.6                    py36_0  
sqlalchemy                1.2.1            py36h14c3975_0  
sqlite                    3.30.1               h7b6447c_0  
statsmodels               0.8.0            py36h8533d0b_0  
sympy                     1.1.1            py36hc6d1c1c_0  
tblib                     1.3.2            py36h34cf8b6_0  
terminado                 0.8.1                    py36_1  
testpath                  0.3.1            py36h8cadb63_0  
tk                        8.6.8                hbc83047_0  
toolz                     0.9.0                    py36_0  
torchvision               0.3.0           py36_cu9.0.176_1    pytorch
tornado                   4.5.3                    py36_0  
traitlets                 4.3.2            py36h674d592_0  
typing                    3.6.2            py36h7da032a_0  
unicodecsv                0.14.1           py36ha668878_0  
unixodbc                  2.3.4                hc36303a_1  
urllib3                   1.22             py36hbe7ace6_0  
wcwidth                   0.1.7            py36hdf4376a_0  
webencodings              0.5.1            py36h800622e_1  
werkzeug                  0.14.1                   py36_0  
wheel                     0.33.6                   py36_0  
widgetsnbextension        3.1.0                    py36_0  
wrapt                     1.10.11          py36h28b7045_0  
xlrd                      1.1.0            py36h1db9f0c_1  
xlsxwriter                1.0.2            py36h3de1aca_0  
xlwt                      1.3.0            py36h7b00a1f_0  
xz                        5.2.4                h14c3975_4  
yaml                      0.1.7                had09818_2  
zeromq                    4.2.2                hbedb6e5_2  
zict                      0.1.3            py36h3a3bf81_0  
zlib                      1.2.11               h7b6447c_3  
<pre>
</details>

<details>
<summary>pip3 list</summary>
<pre>
$ pip3 list
Package                       Version               
----------------------------- ----------------------
alabaster                     0.7.7                 
Babel                         1.3                   
beautifulsoup4                4.4.1                 
blinker                       1.3                   
Brlapi                        0.6.4                 
chardet                       2.3.0                 
command-not-found             0.3                   
cram                          0.6                   
cryptography                  1.2.3                 
cycler                        0.9.0                 
defer                         1.0.6                 
devscripts                    2.16.2ubuntu3         
docutils                      0.12                  
feedparser                    5.1.3                 
html5lib                      0.999                 
httplib2                      0.9.1                 
idna                          2.0                   
Jinja2                        2.8                   
language-selector             0.1                   
louis                         2.6.4                 
lxml                          3.5.0                 
Magic-file-extensions         0.2                   
Mako                          1.0.3                 
MarkupSafe                    0.23                  
matplotlib                    1.5.1                 
numpy                         1.18.1                
oauthlib                      1.0.3                 
onboard                       1.2.0                 
pexpect                       4.0.1                 
Pillow                        7.0.0                 
pip                           19.3.1                
protobuf                      3.11.2                
ptyprocess                    0.5                   
pyasn1                        0.1.9                 
pycups                        1.9.73                
pycurl                        7.43.0                
Pygments                      2.1                   
pygobject                     3.20.0                
PyJWT                         1.3.0                 
pyparsing                     2.0.3                 
python-apt                    1.1.0b1+ubuntu0.16.4.8
python-dateutil               2.4.2                 
python-debian                 0.1.27                
python-systemd                231                   
pytz                          2014.10               
pyxdg                         0.25                  
reportlab                     3.3.0                 
requests                      2.9.1                 
roman                         2.0.0                 
screen-resolution-extra       0.0.0                 
sessioninstaller              0.0.0                 
setuptools                    20.7.0                
six                           1.14.0                
Sphinx                        1.3.6                 
sphinx-rtd-theme              0.1.9                 
ssh-import-id                 5.5                   
system-service                0.3                   
tensorboardX                  2.0                   
terminaltables                3.1.0                 
tqdm                          4.41.1                
ubuntu-drivers-common         0.0.0                 
ufw                           0.35                  
unattended-upgrades           0.1                   
unity-scope-calculator        0.1                   
unity-scope-chromiumbookmarks 0.1                   
unity-scope-colourlovers      0.1                   
unity-scope-devhelp           0.1                   
unity-scope-firefoxbookmarks  0.1                   
unity-scope-gdrive            0.7                   
unity-scope-manpages          0.1                   
unity-scope-openclipart       0.1                   
unity-scope-texdoc            0.1                   
unity-scope-tomboy            0.1                   
unity-scope-virtualbox        0.1                   
unity-scope-yelp              0.1                   
unity-scope-zotero            0.1                   
urllib3                       1.13.1                
usb-creator                   0.3.0                 
virtualenv                    15.0.1                
wheel                         0.29.0                
xdiagnose                     3.8.4.1               
xkit                          0.0.0                 
WARNING: You are using pip version 19.3.1; however, version 20.0.2 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
</pre>
</details>

## 1. 准备数据集

<a href="https://byu.box.com/s/hdgztcu12j7fij397jmd68h4og6ln1jw">比赛的数据集文件</a>下载下来名为`data_training_V4.zip`，解压后名为`data_training`，将其与`dxy_DAC_SDC_2020_model`放在同一目录下。

然后需要将数据集分为训练集（train）和验证集（valid）：
```bash
cd dxy_DAC_SDC_2020_model/dac_sdc_2020

python3 split_data.py
```
默认是按9：1的比例划分train和valid。如果不成功可能需要改一下路径？运行完成后会在`data_training`和`dxy_DAC_SDC_2020_model`同目录下生成`dac_sdc_2020_dataset`文件夹，其中包含`dac.names`（95个类的名字）、`train`（训练集）和`valid`（验证集）

## 2. 配置训练参数

训练参数在`dxy_DAC_SDC_2020_model/train/config.py`中。我没有用命令行传参数，所有的参数配置都在这个里面。看注释应该能看懂，改起来也非常方便。

## 3. 训练

最开始没有预训练，直接随机参数暴力训练。

```bash
cd dxy_DAC_SDC_2020_model/train

python3 dxy_train.py
```
训练时会在config中指定的`working_dir`下面建立文件夹，保存模型和测试结果。每一个`.pth`文件保存了一个字典，其中有config、模型参数、验证集上的结果等。直接`torch.load('.pth')`即可。

每一个epoch的会在验证集上测试一遍，并保存一个`.pth`文件。


# References

- https://github.com/BobLiu20/YOLOv3_PyTorch
- https://zhuanlan.zhihu.com/p/35325884