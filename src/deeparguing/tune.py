import torch
import torch.optim as optim
from deeparguing.gradual_aacbr import GradualAACBR
import ray
from ray import tune
from deeparguing.train import evaluate_model


def objective(config, X_casebase, y_casebase, X_default, y_default, X_train_new_cases, y_train_new_cases,
              X_eval_new_cases, y_eval_new_cases,
              show_confusion=False, print_matrix=False,
              print_compute_graph=False, print_graph=False, print_results=False,
              disable_tqdm = False, plot_loss_curve = False, device="cpu"):

    semantics = config["semantics"]
    partial_order = config["partial_order"]
    irrelevance = config["irrelevance"]
    base_score = config["base_score"]

    model = GradualAACBR(semantics,
                         base_score,
                         irrelevance,
                         partial_order).to(device)
    
    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config["lr"], momentum=config["momentum"])

    config["train"](model, X_casebase, y_casebase, X_default, y_default, optimizer, criterion, config["epochs"], config["use_symmetric_attacks"], 
                    X_new_cases=X_train_new_cases, y_new_cases=y_train_new_cases, n_splits = config.get("n_splits", None), use_blockers=config["use_blockers"], 
                    plot_loss_curve=plot_loss_curve, disable_tqdm=disable_tqdm, random_split_state=config.get("random_split_state", None),
                    regularise_graph = config["regulariser"])

    accuracy, precision, recall, f1 = evaluate_model(model, X_casebase, y_casebase, X_default, y_default, 
                                                     X_eval_new_cases, y_eval_new_cases,
                                                     show_confusion=show_confusion, print_matrix=print_matrix,
                                                     print_compute_graph=print_compute_graph, print_graph=print_graph,
                                                     print_results=print_results, use_symmetric_attacks=config["use_symmetric_attacks"], 
                                                     use_blockers=config["use_blockers"])

    return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}


def tune_model(search_space, X_casebase, y_casebase, X_default, y_default, 
               X_train_new_cases, y_train_new_cases, X_eval_new_cases, y_eval_new_cases,
                disable_tqdm = False, num_cpus=12, num_gpus = 0, num_samples=1000, metric="accuracy", device="cpu"):

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)


    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, X_casebase=X_casebase, y_casebase=y_casebase, X_default=X_default,
                             y_default=y_default, 
                             X_train_new_cases=X_train_new_cases, y_train_new_cases=y_train_new_cases, 
                             X_eval_new_cases = X_eval_new_cases, y_eval_new_cases = y_eval_new_cases,
                             disable_tqdm = disable_tqdm, device=device),
            resources={"cpu": num_cpus, "gpu": num_gpus}
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="max",
            num_samples=num_samples,
        ),
        param_space=search_space
    )

    results = tuner.fit()
    best_params = results.get_best_result().config
    print("BEST PARAMETERS FOR MODEL ARE:")
    print(best_params)
    print("Best score found was: ", results.get_best_result().metrics[metric])
    return best_params

