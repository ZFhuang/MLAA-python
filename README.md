# MLAA-python

A simple python implementation of Morphological Antialiasing

[Notes](https://github.com/ZFhuang/Study-Notes/blob/main/Content/%E5%9B%BE%E5%BD%A2%E5%AD%A6/%E5%BD%A2%E6%80%81%E6%8A%97%E9%94%AF%E9%BD%BFMLAA%E4%B8%8EPython%E5%AE%9E%E7%8E%B0/README.md)

## Requirements

- tqdm
- numpy

## Usage

```shell
python mlaa-cpu.py --origin {your_folder}
```

```shell
optional arguments:
  -h, --help         show this help message and exit
  --origin DIR_ORI   folder of origin images
  --target DIR_TAR   target folder of result images. Default path is
                     "{DIR_ORI}_out/"
  --save STR_SUFFIX  result images suffix
  --type STR_TYPE    type string for finding img edges. "L": using luminance;
                     "P": using per-channel color value
  --th NUM_TH        threshold for finding edges
```

## Figures

### Input

![luminance](./figs/lu.png)

### Edge

![edge](./figs/ed.png)

### Weight

![weight](./figs/we.png)

### Result

![result](./figs/re.png)
