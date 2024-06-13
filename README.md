# ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images [CVPR 2024]
This repository contains the official PyTorch implementation for our CVPR2024 paper.
- [Jinseo Jeong*](https://jinseo.kr/), [Junseo Koo](https://vision.snu.ac.kr/people/junseokoo.html), [Qimeng Zhang](), [Gunhee Kim](https://vision.snu.ac.kr/gunhee/). ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images. In CVPR, 2024 

[[Paper Link]](https://arxiv.org/abs/2404.15707) [[Dataset]](https://www.dropbox.com/scl/fo/wg3p5bm3186oxpecioqsv/AICtbbYSEpLF9HZOGZ-uiE4?rlkey=7k5zt2fwyddfvadshs5h211k9&st=j284d8qp&dl=0)



## Installation
Using virtual env is recommended.
```
$ conda create --name ESR python=3.10
```
Install pytorch, torchvision, and torch-scatter.
Then, install the rest of the requirements.
```
$ pip install -r requirements.txt
```

## Data and Log directory set up
create `logs` and `dataset` directories.
We recommend symbolic links as below.
```
$ mkdir dataset
$ ln -s [ESR-NeRF Data Path] dataset/esrnerf
$ ln -s [DTU Data Path] dataset/dtu

$ ln -s [log directory path] logs
```

## Reconstruction
ESR-NeRF consists of several training stages: alphamask -> coarse -> fine -> lts -> pdra.


Specify an appropriate config file in the `cfg/exp` directory.
The project name and experiment name are up to you.
```
$ python run.py -cn [config file path] app.phase=train log.project=[project name] log.name=[experiment name]

# e.g. to run the giftbox (white) scene,
$ python run.py -cn cfg/exp/esrnerf/giftbox_w/alphamask.yaml app.phase=train log.project=esrnerf log.name=giftbox_w;
$ python run.py -cn cfg/exp/esrnerf/giftbox_w/coarse.yaml app.phase=train log.project=esrnerf log.name=giftbox_w;
$ python run.py -cn cfg/exp/esrnerf/giftbox_w/fine.yaml app.phase=train log.project=esrnerf log.name=giftbox_w;
$ python run.py -cn cfg/exp/esrnerf/giftbox_w/lts.yaml app.phase=train log.project=esrnerf log.name=giftbox_w;
$ python run.py -cn cfg/exp/esrnerf/giftbox_w/pdra.yaml app.phase=train log.project=esrnerf log.name=giftbox_w;



# e.g. to run the dtu 97 scene,
$ python run.py -cn cfg/exp/dtu/97/alphamask.yaml app.phase=train log.project=dtu log.name=97;
$ python run.py -cn cfg/exp/dtu/97/coarse.yaml app.phase=train log.project=dtu log.name=97;
$ python run.py -cn cfg/exp/dtu/97/fine.yaml app.phase=train log.project=dtu log.name=97;
$ python run.py -cn cfg/exp/dtu/97/lts.yaml app.phase=train log.project=dtu log.name=97;
```

## Illumination Decomposition

To render images of decomposed illumination effects, run the below command.
```
# e.g. to run the giftbox scene for rendering scene illumination decomposition,
$ python run.py -cn logs/info/esrnerf/esrnerf.ESRNeRF.giftbox_w.fine.PDRA/giftbox_w/train/cfg.yaml app.phase=test_nv app.eval.render_pbr=True;
```

## Re-lighting
Once the emissive sources are reconstructed, you can edit them.
Re-lighting is achievable with the following command.

```
# Specify a config file in the log directory.
$ python run.py -cn [config file path in the log dir] app.phase=[test type]

# e.g. to run the giftbox scene for color editing,
$ python run.py -cn logs/info/esrnerf/esrnerf.ESRNeRF.giftbox_w.fine.PDRA/giftbox_w/train/cfg.yaml app.phase=test_nvc;

# e.g. to run the cube scene for intensity editing,
$ python run.py -cn logs/info/esrnerf/esrnerf.ESRNeRF.cube_w.fine.PDRA/cube_w/train/cfg.yaml app.phase=test_nvi;

# e.g. to run the book scene for intensity & color editing with 40000 iterations per image,
$ python run.py -cn logs/info/esrnerf/esrnerf.ESRNeRF.book_w.fine.PDRA/book_w/train/cfg.yaml app.phase=test_nvic app.eval.n_iters=40000;
```

### Limitations
Although we demonstrated the plausibility of controlling emissive sources by reconstructing them from LDR images, there are limitations in re-lighting methods.
Re-lighting via radiance fine-tuning is slow and may produce unreliable results (see the Discussion and Appendix in our paper for more details). Also, adjusting the learning rates and iterations may be necessary for the best scene-editing results.

For optimal efficiency, we recommend exporting mesh and texture maps using Blender's SmartUV unwrap algorithm and utilizing physically-based rendering in Blender or Mitsuba for scene editing.

## Citation
The code and dataset are free to use for academic purposes only. If you use any of the material in this repository as part of your work, we ask you to cite:
```
@inproceedings{jeong-CVPR-2024,
    author    = {Jinseo Jeong and Junseo Koo and Qimeng Zhang and Gunhee Kim},
    title     = "{ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images}"
    booktitle = {CVPR},
    year      = 2024
}
```
