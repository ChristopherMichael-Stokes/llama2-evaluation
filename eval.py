import logging
from pathlib import Path
from typing import Any, Dict

import hydra
import papermill as pm


@hydra.main(config_path='conf', config_name='conf.yaml', version_base='1.2.1')
def main(cfg: Dict[str, Any]):
    print(cfg)
    types = cfg['types']
    output_path = cfg['output_path']
    tests = cfg['tests']
    max_samples=cfg['max_samples']

    output = Path(output_path)
    log = logging.getLogger(__name__)

    for test in tests:
        for type_ in types:
            log.info(f'Starting {test} with {type_} model')
            out = pm.execute_notebook(
                f'{test}.ipynb',
                output / f'{test}_{type_}.ipynb',
                parameters={'model_size': type_, 'max_samples': max_samples}
            )

    log.info('SUCCESS')


if __name__=='__main__':
    main()