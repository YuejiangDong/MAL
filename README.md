## :sparkles: Overview

This is the repository for our paper *MAL: Motion-Aware Loss with Temporal and Distillation Hints
for Self-Supervised Depth Estimation*. MAL is a novel, plug-and-play module designed for seamless integration into multi-frame self-supervised monocular depth estimation methods.



## :zap: Quick start

We apply our MAL to three representative self-supervised depth estimation methods to evaluate. We currently provide code for model evaluation in the corresponding directory.

Please refer to the repositories of [ManyDepth](https://github.com/nianticlabs/manydepth) [DualRefine](https://github.com/antabangun/DualRefine/tree/main) and [DynamicDepth](https://github.com/AutoAILab/DynamicDepth/tree/main) for dataset preparing and environment setup. 

To run the evaluation, you should also prepare depth ground truth. For KITTI, please run `export_gt_depth.py` to extract ground truth files. For CityScapes, we use the ground truth depth files provided by ManyDepth. Please download them [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), and unzip into `splits/cityscapes` .

To evaluate a model, run:
|      Method      |  Dataset   |                            Bash                            |
| :--------------: | :--------: | :----------------------------------------------------------- |
| ManyDepth + MAL  |   KITTI    | ```python -m manydepth.evaluate_depth --data_path <your_data_path> --load_weights_folder <model_path>``` |
| ManyDepth + MAL  | CityScapes | ```python -m manydepth.evaluate_depth --data_path <your_data_path> --load_weights_folder <model_path> --eval_cs``` |
| DualRefine + MAL |   KITTI    | ```python -m dualrefine.evaluate_depth --data_path <your_data_path> --load_weights_folder <model_path> --eval_mono``` |
| DualRefine + MAL | CityScapes | ```python -m dualrefine.evaluate_depth --data_path <your_data_path> --load_weights_folder <model_path> --eval_mono --eval_cs``` |
| DynamicDepth + MAL | CityScapes | ```python -m dynamicdepth.train --data_path <your_data_path> --load_weights_folder <model_path> --mono_weights_folder <model_path> --eval_mode``` |




## :file_folder: Models

|      Method      |  Dataset   |                            Models                            |
| :--------------: | :--------: | :----------------------------------------------------------: |
| ManyDepth + MAL  |   KITTI    | [many_k](https://drive.google.com/drive/folders/1seIentzd44FOSbA29rYDNrRcB4CSVzPt?usp=drive_link) |
| ManyDepth + MAL  | CityScapes | [many_cs](https://drive.google.com/drive/folders/1TWXTTENZZA_1yuhBIweGvjQoPLS6Jvun?usp=drive_link) |
| DualRefine + MAL |   KITTI    | [dual_k](https://drive.google.com/drive/folders/10vtBMZCU8OxC8QD4wKu2ur0oVjsdHAjz?usp=drive_link) |
| DualRefine + MAL | CityScapes | [dual_cs](https://drive.google.com/drive/folders/1lR_TKoYyUGqgvj37WafRE67maXn42UOO?usp=drive_link) |
| DynamicDepth + MAL | CityScapes | [dyn_cs](https://drive.google.com/drive/folders/1ggh_Kfxq5c0m7EjqA5oK3REIb7X4VmxH?usp=share_link) |

