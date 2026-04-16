import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import importlib.util
import logging

from internnav.evaluator import Evaluator

# This file is the main file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_rdp_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress debug/info logs and show only progress + metrics on console",
    )
    return parser.parse_args()


def apply_quiet_mode():
    import os
    from datetime import datetime

    from internnav import PROJECT_ROOT_PATH
    from internnav.utils import progress_log_multi_util

    # --- file handler + print redirect (added only once) ---
    if not hasattr(apply_quiet_mode, '_file_handler'):
        import builtins

        log_dir = os.path.join(PROJECT_ROOT_PATH, 'logs', 'eval')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        log_fp = open(log_file, 'a')  # noqa: WPS515

        # route logging records to the file
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(filename)s[line:%(lineno)d] -: %(message)s'))
        logging.root.addHandler(fh)
        logging.root.setLevel(logging.DEBUG)  # allow all levels to reach the file handler
        apply_quiet_mode._file_handler = fh

        # redirect print() to the same file (agent uses print, not logging)
        _orig_print = builtins.print

        def _quiet_print(*args, **kwargs):
            kwargs.setdefault('file', log_fp)
            _orig_print(*args, **kwargs)
            log_fp.flush()

        builtins.print = _quiet_print
        apply_quiet_mode._orig_print = _orig_print
        apply_quiet_mode._log_fp = log_fp

        _orig_print(f'[quiet] detailed logs -> {log_file}')

    # --- suppress console StreamHandlers on all existing loggers ---
    _our_handlers = {getattr(apply_quiet_mode, '_file_handler', None), getattr(apply_quiet_mode, '_console_handler', None)}

    def _suppress(logger):
        for h in logger.handlers:
            if h in _our_handlers:
                continue
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.WARNING)

    _suppress(logging.root)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            _suppress(logger)

    # --- keep progress visible on console (add handler only once) ---
    if not hasattr(apply_quiet_mode, '_console_handler'):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        progress_log_multi_util.progress_logger_multi.setLevel(logging.INFO)
        progress_log_multi_util.progress_logger_multi.propagate = False
        progress_log_multi_util.progress_logger_multi.addHandler(console)
        apply_quiet_mode._console_handler = console


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def main():
    args = parse_args()
    if args.quiet:
        apply_quiet_mode()
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')

    # fill in evaluator default config
    if evaluator_cfg.eval_type == 'vln_distributed':
        from internnav.configs.evaluator.vln_default_config import get_config

        evaluator_cfg = get_config(evaluator_cfg)

    # create evaluator based on sim backend and run eval
    evaluator = Evaluator.init(evaluator_cfg)
    if args.quiet:
        apply_quiet_mode()  # re-apply after Isaac Sim init (it adds new console handlers)
    evaluator.eval()


if __name__ == '__main__':
    main()
