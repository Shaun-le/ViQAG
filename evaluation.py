import fire
from plms.model_evaluation_qag import Evaluation
from plms.compute_metrics import Evaluate
class QAGenerationEvaluation:
    def generate(
        self,
        model: str = 'VietAI/vit5-base',
        model_ae: str = None,
        max_length: int = 512,
        max_length_output: int = 256,
        dataset_path: str = 'shnl/qg-example',
        dataset_name: str = '',
        test_split: str = 'test',
        validation_split: str = 'validation',
        n_beams: int = 8,
        batch_size: int = 4,
        language: str = 'vi',
        use_auth_token: bool = True,
        device_map: str = None,
        low_cpu_mem_usage: bool = False,
        export_dir: str = './result',
        hyp_test: str = None,
        hyp_dev: str = None,
        overwrite_prediction: bool = True,
        overwrite_metric: bool = True,
        is_qg: bool = None,
        is_ae: bool = None,
        is_qag: bool = True,
        use_reference_answer: bool = False
    ):
        assert (
            model
        ), "Please specify your model, default. --model='VietAI/vit5-base'"
        eval = Evaluation(
            model = model,
            model_ae = model_ae,
            max_length = max_length,
            max_length_output = max_length_output,
            dataset_path = dataset_path,
            dataset_name = dataset_name,
            test_split = test_split,
            validation_split = validation_split,
            n_beams = n_beams,
            batch_size = batch_size,
            language = language,
            use_auth_token = use_auth_token,
            device_map = device_map,
            low_cpu_mem_usage = low_cpu_mem_usage,
            export_dir = export_dir,
            hyp_test = hyp_test,
            hyp_dev = hyp_dev,
            overwrite_prediction = overwrite_prediction,
            overwrite_metric = overwrite_metric,
            is_qg = is_qg,
            is_ae = is_ae,
            is_qag = is_qag,
            use_reference_answer = use_reference_answer
        )
        eval.evaluation()

    def evaluate(
        self,
        result_path: str = ''
    ):
        assert (
            result_path
        ), "result_path cannot be empty."
        evaluator = Evaluate(result_path)
        evaluator.compute_metrics()

if __name__ == "__main__":
    qageneration_evaluation = QAGenerationEvaluation()
    fire.Fire(qageneration_evaluation)