import argparse
from MIMIC_Outcomes import run_analyses, run_tasks

def get_args():
    par = argparse.ArgumentParser(description="MIMIC runs")
    par.add_argument('-input_path', type=str, required=True, help='input path')
    par.add_argument('-output_path',type=str, required=True, help='output path')    
    par.add_argument('-feats_path', type=str, required=True, help='features path')
    par.add_argument('-cache_path', type=str, help='tmp path')
    par.add_argument('-dataset', type=str, required=True, help='dataset')
    par.add_argument('-feature_type', type=str, required=True, help='feature type')
    par.add_argument('-clear_results', action="store_true", help='clear results cache')    
    par.add_argument('-tune', type=str, help='tuning metric')
    par.add_argument('-mini_tasks', action="store_true", help='use small datasets')
    par.add_argument('-reset_tasks', action="store_true", help='re-run all tasks')
    par.add_argument('-fairness', action="store_true", help='optimize random seed for fairness')
    
    return par.parse_args()  	

if __name__ == "__main__":
    args = get_args()    
    if "tasks" in args.dataset:
        print("[running tasks: {}]".format(args.dataset))
        print("input:{}\nfeature_type:{}\noutput:{}\ncache:{}\nclear_results:{}\ntune:{}".format(args.input_path, args.feature_type, args.output_path, args.cache_path, args.clear_results, args.tune))
        print("mini_tasks:{}\nreset_tasks:{}".format(args.mini_tasks, args.reset_tasks))        
        run_tasks(args.input_path, args.dataset+".txt", args.feats_path, args.feature_type, 
        args.output_path, args.cache_path, args.reset_tasks, args.tune, None, args.mini_tasks,
        fairness=args.fairness)    
    else:        
        print("input:{}\ndataset:{}\nfeature_type:{}\noutput:{}\ncache:{}\nclear_results:{}\ntune:{}\n".format(args.input_path, args.dataset, args.feature_type, args.output_path, args.cache_path, args.clear_results, args.tune))
        run_analyses(args.input_path, args.dataset, args.feats_path, args.feature_type, 
                    args.output_path, args.cache_path, args.clear_results, args.tune,
                    fairness=args.fairness)
