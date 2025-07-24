<p align="center">
  <h1 align="center">VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback</h1>
</p>


[![pytorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jxbi1010/VLA-Touch/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/ArXiv-2410.07584-b31b1b.svg)](https://arxiv.org/abs/2507.17294)

<!-- [![ICRA 2025](https://img.shields.io/badge/ICRA%202025-Accepted-purple.svg)](https://icra2025.org) -->

This repo contains the code implementation for VLA-Touch:

<p align="center">&nbsp;<table><tr><td>
    <p align="center">
    <strong>
        <a href="https://arxiv.org/abs/2507.17294">
            VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback
        </a><br/>
    </strong>
    Jianxin Bi <sup>1</sup>, Kevin Yuchen Ma <sup>1,3</sup>, Ce Hao <sup>1</sup>, Mike Zheng Shou <sup>1,3</sup>, Harold Soh <sup>1,2</sup><br>
    <sup>1</sup><em>Dept. of Computer Science, National University of Singapore</em><br>
    <sup>2</sup><em>Smart Systems Institute, NUS</em><br>
    <sup>3</sup><em>Show Lab, NUS</em>
</td></tr></table>&nbsp;


# 🧾 Introduction


We present **VLA-Touch**, an approach that enhances generalist robot policies with tactile sensing *without fine-tuning* the base VLA. Our method introduces two key innovations: (1) a pipeline that leverages a pretrained tactile-language model that provides semantic tactile feedback for high-level task planning, and (2) a diffusion-based controller that refines VLA-generated actions with tactile signals for contact-rich manipulation. Through real-world experiments, we demonstrate that our dual-level integration of tactile feedback improves task planning efficiency while enhancing execution precision. 

<div align="center">
  <img src="assets/framework.jpg" alt="VLA-Touch Framework" width="700">
</div>


Figure 1:Dual-level Tactile feedback framework of VLA-Touch. **Planning**: Given a scene image $s_t$ and task goal $g$, the VLM Task Planner generates manipulation instruction $I_k$ for policy execution. A tactile-language model (Octopi) converts a sequence tactile input $o^m_{t-n:t}$ to language description $L^m_t$, which informs VLM for updated instruction. **Manipulation**: The base VLA $\pi(a_t|s_t,I_k)$ generates action chunk $a_t$ based on visual observation $s_t$ and instruction $I_k$. The action chunk is then refined by an interpolant policy $\pi_I(\hat a_t|s_t,a_t,m_t)$ that takes as input both visual embeddings from a pretrained DinoV2 model and low-dimensional tactile signals $m_t$ processed a marker tracking algorithm from raw tactile input $o^m_t$.




# 💻 Installation

1. **Follow RDT-1B installation**  
   See the official instructions:  [RDT-1B installation](https://github.com/thu-ml/RoboticsDiffusionTransformer)

2. **Clone VLA-Touch and copy files to RDT-1B (replace original files):**
   ```bash
   git clone https://github.com/jxbi1010/VLA-Touch
   # Copy relevant files to your RDT-1B directory, replacing originals as needed
   ```

3. **Download dataset and controller checkpoints:**  [Google Drive Folder](https://drive.google.com/drive/folders/1k_tGMJVIhZX6KHRa0SRjM73hvHaVEXvW?usp=sharing)

   - Copy controller checkpoints:
     ```bash
     cp controller_ckpt/* VLA/residual_controller/checkpoints/
     ```

4. **Dataset processing:**

   - Copy dataset files:
     ```bash
     cp vla_data/* VLA/data/datasets/
     ```

   - Convert raw data to `.h5` format:
     ```bash
     # Run the provided scripts to convert raw data
     cd VLA/data/franka_data
     python convert*_to_h5.py  # Replace with actual processing scripts
     # The resulting files should look like: vla_data/wipe_example/episode_*.h5
     ```

   - *If you need our processed dataset, kindly approach us.*

5. **Compute dataset stats and update configs:**
   ```bash
   # Use RDT scripts to compute dataset statistics
   python compute_dataset_stats.py 
   ```

6. **Install Octopi:**  
   Follow the instructions in:
   ```
   octopi/README.md
   ```

7. **Copy Octopi data files:**
   ```bash
   # Download from Google Drive and copy to the correct location
   cp octopi_data/* octopi/octopi_s/data/
   ```
   [Google Drive Folder for Octopi Data](https://drive.google.com/drive/folders/1k_tGMJVIhZX6KHRa0SRjM73hvHaVEXvW?usp=sharing)


# 🛠️ Usage
1. Follow RDT-1B for VLA base model finetuning without tactile data.
2. Run scripts in residual_controller/ for controller training and test, e.g.

```bash
# training for interpolant controller
python bridge_train.py

# testing for interpolant controller
python bridger_test.py

#training for residual controller
python lstm_train.py

# testing for residual controller
python lstm_step_test.py
```

3. For ocpoti inference, run octopi/octopi_s/touch_vla.py using your own VLM API.
4. Inference method is modified based on RDT inference script, our version will release soon.


# 📝 Citation

If you find our work useful, please consider citing:
```bibtex
@misc{bi2025vlatouchenhancingvisionlanguageactionmodels,
      title={VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback}, 
      author={Jianxin Bi and Kevin Yuchen Ma and Ce Hao and Mike Zheng Shou and Harold Soh},
      year={2025},
      eprint={2507.17294},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.17294}, 
}
``` 

<br></br>
# 🏷️ License
**VLA-Touch** is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

<br></br>
# 🙏 Acknowledgement

**VLA-Touch** is developed based on many open-sourced works, including [BRIDGeR](https://github.com/clear-nus/bridger), [Octopi](https://github.com/clear-nus/octopi) and [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer). We thank all these authors for their nicely open sourced code and their great contributions to the community.