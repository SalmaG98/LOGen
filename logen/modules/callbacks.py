import os
import subprocess
import traceback
import json
from pytorch_lightning.callbacks.callback import Callback
from typing_extensions import override
import numpy as np
import torch

class GenerationEvalCallback(Callback):
    def __init__(
        self,
        project_root: str,
        gen_script: str = "scripts/gen.sh",
        eval_scripts_dir: str = "evaluation",
        run_every_epochs: int = 1,
        gen_args: list | None = None,
        eval_args: dict | None = None,
    ):
        super().__init__()
        self.project_root = project_root
        self.gen_script = os.path.join(project_root, gen_script)
        self.eval_scripts_dir = os.path.join(project_root, eval_scripts_dir)
        self.run_every_epochs = run_every_epochs
        self.gen_args = gen_args or []
        # eval_args is a mapping: {"eval_foo.sh": ["--arg1", "x"], ...}
        self.eval_args = eval_args or {}
        self.jobs = {}

    def _run_cmd_async(self, cmd, cwd=None):
        try:
            result = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            print("Failed to launch:", cmd)
            traceback.print_exc()
            return None
        
        return result

    def _parse_metrics_from_output(self, output: str) -> dict:
        # all eval metric .py scripts return a json dict, \
        # callback must be call scripts with --silent flag to avoid breaking json parsin
        return json.loads(output)

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking and trainer.is_global_zero:
            print(f"GEN-EVAL CALLBACK STARTED, callback attatched to rank {torch.distributed.get_rank()}")

            # SG(BUG): because pytorch lightining is reordering callbacks and calling checkpoint saving \
            # after eval callback we mush evaluate using the previous validation epoch
            epoch = trainer.current_epoch - 1
            # Run only every N epochs
            if epoch % self.run_every_epochs != 0:
                return

            ckpt_path = trainer.checkpoint_callback.last_model_path
            if not ckpt_path:
                print(f"[GEN-EVAL CALLBACK] EPOCH {epoch}: no checkpoint saved yet")
                return

            # 1) Run generation script
            print(f"[GEN-EVAL CALLBACK] Launching generation for epoch {epoch}")
            gen_cmd = ["bash", self.gen_script] + self.gen_args + [f"{epoch}-last"]
            gen_proc = self._run_cmd_async(gen_cmd, cwd=self.project_root)

            if gen_proc is not None:
                self.jobs[epoch] = {"gen": gen_proc, "eval": None}

            finished_epochs = []

            for ep, job in self.jobs.items():

                gen_proc = job["gen"]
                eval_procs = job["eval"]

                # if no eval process yet check if generation is done
                if eval_procs is None:

                    ret = gen_proc.poll()

                    if ret is None:
                        # generation is still running
                        continue

                    stdout, stderr = gen_proc.communicate()

                    if ret != 0:
                        print(f"[GEN-EVAL][WARNING] Generation failed for epoch {ep}")
                        print(stderr)
                        finished_epochs.append(ep)
                        continue

                    print(f"[GEN-EVAL] Generation finished for epoch {ep}")

                    eval_procs = {}

                    for idx, (script_name, extra_args) in enumerate(self.eval_args.items()):
                        script_path = os.path.join(self.eval_scripts_dir, script_name)

                        cmd = ["bash", script_path] + extra_args + [f"epoch_{ep}", "True"]

                        proc = self._run_cmd_async(cmd, cwd=self.project_root)

                        if proc is not None:
                            eval_procs[idx] = proc

                    job["eval"] = eval_procs
                    continue
                all_finished = True

                for proc in eval_procs.values():
                    if proc.poll() is None:
                        all_finished = False
                        break

                if not all_finished:
                    continue

                print(f"[GEN-EVAL] All eval scripts finished for epoch {ep}")

                all_metrics = {}

                for idx, proc in eval_procs.items():

                    stdout, stderr = proc.communicate()

                    if proc.returncode != 0:
                        print(f"[WARNING] Eval script {idx} failed for epoch {ep}")
                        print(stderr)
                        continue

                    try:
                        metrics = self._parse_metrics_from_output(stdout)
                    except Exception:
                        print(f"[WARNING] Failed parsing JSON for eval {idx}")
                        traceback.print_exc()
                        continue

                    for name, value in metrics.items():
                        all_metrics[name] = value

                # Log metrics
                for name, value in all_metrics.items():
                    pl_module.log(
                        f"eval/{ep}/{name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                    )

                finished_epochs.append(ep)
            
            for ep in finished_epochs:
                del self.jobs[ep]