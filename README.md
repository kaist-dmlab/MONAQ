# 🫅🏻 MONAQ
This is the official implementation of the _MONAQ: Multi-Objective Neural Architecture Querying for Time-Series Analysis on Resource-Constrained Devices_ paper [[arXiv](https://arxiv.org/abs/2505.10607)] [[Poster](static/pdfs/poster.pdf)].
<img width="1243" alt="image" src="https://github.com/user-attachments/assets/783f8a95-4e55-446a-9b53-d64f6cb9f213" />

## Abstract   
The growing use of smartphones and IoT devices necessitates efficient time-series analysis on resource-constrained hardware, which is critical for sensing applications such as human activity recognition and air quality prediction. Recent efforts in hardware-aware neural architecture search (NAS) automate architecture discovery for specific platforms; however, none focus on general time-series analysis with edge deployment. Leveraging the problem-solving and reasoning capabilities of large language models (LLM), we propose ***MONAQ***, a novel framework that reformulates NAS into ***M***ulti-***O***bjective ***N***eural ***A***rchitecture ***Q***uerying tasks. *MONAQ* is equipped with *multimodal query generation* for processing multimodal time-series inputs and hardware constraints, alongside an *LLM agent-based multi-objective search* to achieve deployment-ready models via code generation. By integrating numerical data, time-series images, and textual descriptions, *MONAQ* improves an LLM's understanding of time-series data. Experiments on fifteen datasets demonstrate that *MONAQ*-discovered models outperform both handcrafted models and NAS baselines while being more efficient.

## Citation
```bibtex 
@misc{trirat2025monaq,
      title={MONAQ: Multi-Objective Neural Architecture Querying for Time-Series Analysis on Resource-Constrained Devices}, 
      author={Patara Trirat and Jae-Gil Lee},
      year={2025},
      eprint={2505.10607},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10607}, 
}
```      
