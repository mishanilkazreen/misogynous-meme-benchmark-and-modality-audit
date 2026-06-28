# pylint: disable=subprocess-run-check
import os
import subprocess

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))


new_entries = """
@article{kim2026vision,
  author    = {Kim, S. and Oh, Y. and Oh, J. and Lee, I.},
  title     = {Vision-Language Target Retrieval for Drone-Based Maritime Search and Rescue},
  journal   = {Korean Journal of Remote Sensing},
  year      = {2026},
  url       = {https://www.kjrs.org/journal/view.html?pn=related&uid=1186&vmd=Full}
}

@article{xu2026navigating,
  author    = {Xu, J. and Yang, Q. and Yang, Z. and Gao, Y. and Yang, K.},
  title     = {Navigating Maritime Emergencies with Large Models: A Lifecycle Oriented Review},
  journal   = {SSRN Electronic Journal},
  year      = {2026},
  url       = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6365328}
}

@article{wu2026trainingfree,
  author    = {Wu, J. and Chen, Y. and Chen, W. and Lai, Y. and Li, J. and Chen, X. and Wu, W.},
  title     = {A Training-Free Paradigm for Data-Scarce Maritime Scene Classification Using Vision-Language Models},
  journal   = {Sensors},
  year      = {2026},
  url       = {https://www.mdpi.com/1424-8220/26/8/2549}
}

@inproceedings{yuan2024patrol,
  author    = {Yuan, Z. and Xie, F. and Ji, T.},
  title     = {Patrol Agent: An Autonomous UAV Framework for Urban Patrol Using On-Board Vision Language Model and On-Cloud Large Language Model},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Computer Vision (ICRCV)},
  year      = {2024},
  url       = {https://ieeexplore.ieee.org/abstract/document/10758606/}
}

@article{ma2026design,
  author    = {Ma, J. and Wang, J. and Yin, H. and Su, X. and Tian, Y. and Deng, T.},
  title     = {Design and System Implementation of an Active Perception Architecture for Maritime Targets Based on the Cognitive Mechanism of Large Models},
  journal   = {IEEE Sensors Journal},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/abstract/document/11478634/}
}

@inproceedings{kassem2026uavbased,
  author    = {Kassem, M. and Abusirdaneh, M. and Talib, M. A.},
  title     = {UAV-Based Wildlife Detection using Deep Learning and Resource-Constrained Edge Devices},
  booktitle = {Proceedings of the International Conference on Resource-Constrained Devices},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/abstract/document/11541524/}
}
"""


def main():
    if not os.path.exists(bib_path):
        print(f"Error: {bib_path} not found")
        return

    print(f"Appending new entries to {bib_path}...")
    with open(bib_path, "a", encoding="utf-8") as f:
        f.write("\n" + new_entries.strip() + "\n")

    print("Running clean_references.py to process and format the new entries...")
    result = subprocess.run(
        ["python3", "scripts/clean_references.py"], cwd="[redacted-path]", capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    main()
