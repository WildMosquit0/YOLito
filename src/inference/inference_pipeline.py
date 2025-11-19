from typing import Any, Dict
from src.inference.inferer import Inferer
from src.postprocess.saver import ResultsParser
from src.utils.sahi_usage import sahi_usage  # For SAHI-based inference


def run_inference(config: Dict[str, Any]) -> None:
    inferer = Inferer(config)
    results = inferer.infer(persist=True)
    parser = ResultsParser(results=results, config=config)

    # For SAHI-based tasks
    if config['model']['task'] == 'slice':
        # Pass the already-obtained SAHI results
        parser.parse_and_save_slice(results)
        


    else:
        parser.parse_and_save()
