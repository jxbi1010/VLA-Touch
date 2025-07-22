<p align="center">
  <h1 align="center">VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback</h1>
</p>


<!-- [![pytorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jxbi1010/KOAP/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/ArXiv-2410.07584-b31b1b.svg)](https://arxiv.org/abs/2410.07584)
[![ICRA 2025](https://img.shields.io/badge/ICRA%202025-Accepted-purple.svg)](https://icra2025.org) -->

<!-- This repo will release the code implementation for VLA-Touch:

<p align="center">&nbsp;<table><tr><td>
    <p align="center">
    <strong>
        <a href="https://arxiv.org/abs/2410.07584">
            VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback
        </a><br/>
    </strong>
    Jianxin Bi <sup>1</sup>, Kevin Ma <sup>1</sup>, Ce Hao <sup>1</sup>, Mike Zheng Shou <sup>1</sup>, Harold Soh <sup>1,2</sup><br>
    <sup>1</sup><em>Department of Computer Science, National University of Singapore</em><br>
    <sup>2</sup><em>Smart System Institute, NUS</em>
</td></tr></table>&nbsp; -->


# 🧾 Introduction


We present **VLA-Touch**, an approach that enhances generalist robot policies with tactile sensing *without fine-tuning* the base VLA. Our method introduces two key innovations: (1) a pipeline that leverages a pretrained tactile-language model that provides semantic tactile feedback for high-level task planning, and (2) a diffusion-based controller that refines VLA-generated actions with tactile signals for contact-rich manipulation. Through real-world experiments, we demonstrate that our dual-level integration of tactile feedback improves task planning efficiency while enhancing execution precision. 

<!-- <div align="center">
  <img src="assets/teaser.jpg" alt="VLA-Touch Framework" width="700">
</div>


Figure 1: Overview of VLA-Touch. <b>Left:</b> Tactile-Assisted Task Planning—The VLM task planner actively acquires tactile feedback; Octopi interprets contacted objects and generates linguistic tactile descriptions to inform subsequent plans. <b>Right:</b> Tactile-Enhanced Manipulation—The Interpolant Model refines VLA-generated actions using tactile signals, enabling improved contact-rich interactions (e.g., more consistent contact with the mango surface during peeling). -->



<div align="center">
  <img src="assets/framework.jpg" alt="VLA-Touch Framework" width="700">
</div>


Figure 1:Dual-level Tactile feedback framework of VLA-Touch. **Planning**: Given a scene image $s_t$ and task goal $g$, the VLM Task Planner generates manipulation instruction $I_k$ for policy execution. A tactile-language model (Octopi) converts a sequence tactile input $o^m_{t-n:t}$ to language description $L^m_t$, which informs VLM for updated instruction. **Manipulation**: The base VLA $\pi(a_t|s_t,I_k)$ generates action chunk $a_t$ based on visual observation $s_t$ and instruction $I_k$. The action chunk is then refined by an interpolant policy $\pi_I(\hat a_t|s_t,a_t,m_t)$ that takes as input both visual embeddings from a pretrained DinoV2 model and low-dimensional tactile signals $m_t$ processed a marker tracking algorithm from raw tactile input $o^m_t$.


# 💻 Code

Code will be released soon.


<br> </br>
# 🙏 Acknowledgement

**VLA-Touch** is developed based on many open-sourced works, including [BRIDGeR](https://github.com/clear-nus/bridger), [Octopi](https://github.com/clear-nus/octopi) and [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer). We thank all these authors for their nicely open sourced code and their great contributions to the community.


<br> </br>
# 🏷️ License
**VLA-Touch** is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.





<!-- # 💻 Installation

Install KOAP and D3IL Benchmark:
```bash
# Clone KOAP
git clone https://github.com/jxbi1010/KOAP

# Install D3IL benchmark from the official repository
cd KOAP/src/environments
git clone https://github.com/ALRhub/d3il
cd d3il
pip install -e .
```

Follow `environments/d3il/README.md` to register gym environment.


```bash
# Install Vector-Quantization package for baseline methods:
pip install vector-quantize-pytorch

# Install other dependencies:
pip install -r requirements.txt
```

Download the dataset to `environments/dataset/data/` following the D3IL benchmark instructions.
```bash
# Generate observation dataset for training
python create_small_dataset.py
```

# 🛠️ Usage

To reproduce our experimental results, run the following commands:

```bash
# Train and evaluate KOAP method
python run_script_koap.py

# Train and evaluate baseline methods
python run_script_<method>.py
```

Replace `<method>` with the specific baseline method you want to run.




# 📝 Citation

If you find our work useful, please consider citing:
```bibtex
@misc{bi2025imitationlearninglimitedactions,
      title={Imitation Learning with Limited Actions via Diffusion Planners and Deep Koopman Controllers}, 
      author={Jianxin Bi and Kelvin Lim and Kaiqi Chen and Yifei Huang and Harold Soh},
      year={2025},
      eprint={2410.07584},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.07584}, 
}
``` -->